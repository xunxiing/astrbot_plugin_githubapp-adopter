from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from cryptography.hazmat.primitives import serialization

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain
from astrbot.api.platform import (
    AstrBotMessage,
    MessageMember,
    MessageType,
    Platform,
    PlatformMetadata,
    register_platform_adapter,
)
from astrbot.core.platform.astr_message_event import MessageSesion
from astrbot.core.utils.astrbot_path import (
    get_astrbot_config_path,
    get_astrbot_plugin_data_path,
)
from astrbot.core.utils.webhook_utils import log_webhook_info

from .github_event import SUPPORTED_EVENTS, GitHubParsedEvent, parse_github_event
from .github_event_message import GitHubAppMessageEvent
from .security import (
    DeliveryDeduplicator,
    fallback_delivery_id,
    verify_github_signature,
)

PLUGIN_ROOT_DIR = "astrbot_plugin_githubapp-adopter"


class PluginConfigStore:
    def __init__(self, plugin_root_dir: str) -> None:
        config_root = Path(get_astrbot_config_path())
        self._path = config_root / f"{plugin_root_dir}_config.json"
        self._cached_data: dict[str, Any] = {}
        self._cached_mtime: float = -1.0

    def get(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return {}
        if mtime == self._cached_mtime:
            return self._cached_data
        try:
            with self._path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self._cached_data = data
                    self._cached_mtime = mtime
                    return data
        except Exception as exc:
            logger.warning(f"[GitHubApp] failed to load plugin config: {exc}")
        return self._cached_data


def _ensure_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


def _make_message_chain_text(message_chain: MessageChain) -> str:
    parts: list[str] = []
    for component in message_chain.chain:
        if isinstance(component, Plain):
            parts.append(component.text)
        else:
            parts.append(f"[{component.type}]")
    return "".join(parts).strip()


def _normalize_pem_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    value = text.strip()
    if not value:
        return ""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _is_valid_pem_private_key(pem: str) -> bool:
    try:
        serialization.load_pem_private_key(
            pem.encode("utf-8"),
            password=None,
        )
        return True
    except Exception:
        return False


def _read_first_valid_private_key_text(paths: list[str]) -> str:
    for path in paths:
        if not path:
            continue
        target_path = Path(path)
        if not target_path.is_file():
            continue
        try:
            pem = target_path.read_text(encoding="utf-8")
            pem = _normalize_pem_text(pem)
            if pem and _is_valid_pem_private_key(pem):
                return pem
        except Exception:
            continue
    return ""


@register_platform_adapter(
    "github_app",
    "GitHub App Webhook 适配器（AstrBot）",
    default_config_tmpl={
        "id": "github_app",
        "type": "github_app",
        "enable": False,
        "github_app_id": "",
        "github_webhook_secret": "",
        "github_api_base_url": "https://api.github.com",
        "github_events": [],
        "wake_event_types": ["issues", "pull_request"],
        "github_signature_validation": True,
        "github_delivery_cache_ttl_seconds": 900,
        "github_delivery_cache_max_entries": 10000,
        "unified_webhook_mode": True,
        "webhook_uuid": "",
    },
    adapter_display_name="GitHub 应用",
    support_streaming_message=False,
)
class GitHubAppAdapter(Platform):
    def __init__(
        self,
        platform_config: dict,
        platform_settings: dict,
        event_queue: asyncio.Queue,
    ) -> None:
        super().__init__(platform_config, event_queue)
        self.settings = platform_settings
        self._shutdown_event = asyncio.Event()

        platform_id = cast(str, platform_config.get("id", "github_app"))
        self._metadata = PlatformMetadata(
            name="github_app",
            description="GitHub App Webhook 适配器（AstrBot）",
            id=platform_id,
            support_streaming_message=False,
        )

        self._plugin_config_store = PluginConfigStore(PLUGIN_ROOT_DIR)
        self._delivery_cache = DeliveryDeduplicator()

        self.github_app_id = ""
        self.github_webhook_secret = ""
        self.github_api_base_url = "https://api.github.com"
        self.enable_signature_validation = True
        self.github_events: set[str] = set(SUPPORTED_EVENTS)
        self.wake_event_types: set[str] = set()
        self.private_key_text = ""

        self._refresh_runtime_config()

    def meta(self) -> PlatformMetadata:
        return self._metadata

    async def send_by_session(
        self,
        session: MessageSesion,
        message_chain: MessageChain,
    ):
        text = _make_message_chain_text(message_chain)
        if text:
            logger.info(
                f"[GitHubApp] outgoing message ignored (session={session.session_id}): {text[:200]}"
            )
        await super().send_by_session(session, message_chain)

    def run(self):
        return self._run()

    async def _run(self):
        if self.unified_webhook():
            log_webhook_info(
                self.meta().id + "(GitHub App)", self.config["webhook_uuid"]
            )
        else:
            logger.warning(
                "[GitHubApp] unified_webhook_mode is disabled or webhook_uuid missing"
            )
        await self._shutdown_event.wait()

    async def terminate(self):
        self._shutdown_event.set()

    async def webhook_callback(self, request: Any) -> Any:
        self._refresh_runtime_config()

        raw_body = cast(bytes, await request.get_data())
        event_name = str(request.headers.get("X-GitHub-Event", "")).strip().lower()
        delivery_id = str(request.headers.get("X-GitHub-Delivery", "")).strip()
        signature = request.headers.get("X-Hub-Signature-256")

        if not raw_body:
            return {"error": "empty body"}, 400
        if not event_name:
            return {"error": "missing event header"}, 400

        if self.enable_signature_validation:
            if not self.github_webhook_secret:
                return {"error": "webhook secret missing"}, 500
            if not verify_github_signature(
                self.github_webhook_secret, raw_body, signature
            ):
                return {"error": "invalid signature"}, 401

        dedup_key = (
            f"{event_name}:{delivery_id}"
            if delivery_id
            else fallback_delivery_id(event_name, raw_body)
        )
        if await self._delivery_cache.seen_before(dedup_key):
            return {"status": "duplicate_ignored"}, 200

        if event_name == "ping":
            return {"status": "pong"}, 200

        if event_name not in self.github_events:
            return {"status": "ignored", "reason": "event_not_subscribed"}, 200

        payload = await request.get_json(silent=True)
        if not isinstance(payload, dict):
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                return {"error": "invalid json"}, 400
        if not isinstance(payload, Mapping):
            return {"error": "invalid payload"}, 400

        parsed = parse_github_event(event_name, payload)
        if not parsed:
            return {"status": "ignored", "reason": "unsupported_event"}, 200

        event = self._build_message_event(parsed, payload, delivery_id)
        if parsed.event_name in self.wake_event_types:
            event.is_wake = True
            event.is_at_or_wake_command = True

        self.commit_event(event)
        return {"status": "accepted"}, 200

    def _refresh_runtime_config(self) -> None:
        plugin_cfg = self._plugin_config_store.get()

        self.github_app_id = str(self.config.get("github_app_id", "")).strip()
        self.github_webhook_secret = str(
            self.config.get("github_webhook_secret", "")
        ).strip()
        self.github_api_base_url = str(
            self.config.get("github_api_base_url", "https://api.github.com")
        ).rstrip("/")

        signature_validation = self.config.get("github_signature_validation")
        if signature_validation is None:
            signature_validation = plugin_cfg.get("enable_signature_validation", True)
        self.enable_signature_validation = bool(signature_validation)

        events = _ensure_list(self.config.get("github_events"))
        if not events:
            events = _ensure_list(plugin_cfg.get("default_github_events"))
        if events:
            self.github_events = {e for e in events if e in SUPPORTED_EVENTS}
        else:
            self.github_events = set(SUPPORTED_EVENTS)

        wake_events = _ensure_list(self.config.get("wake_event_types"))
        if not wake_events:
            wake_events = _ensure_list(plugin_cfg.get("default_wake_event_types"))
        self.wake_event_types = {e for e in wake_events if e in SUPPORTED_EVENTS}

        ttl_seconds = self.config.get(
            "github_delivery_cache_ttl_seconds",
            plugin_cfg.get("delivery_cache_ttl_seconds", 900),
        )
        max_entries = self.config.get(
            "github_delivery_cache_max_entries",
            plugin_cfg.get("delivery_cache_max_entries", 10000),
        )
        self._delivery_cache.reconfigure(
            int(ttl_seconds) if str(ttl_seconds).strip() else 900,
            int(max_entries) if str(max_entries).strip() else 10000,
        )

        private_key_paths = self._resolve_private_key_paths(plugin_cfg)
        self.private_key_text = _read_first_valid_private_key_text(private_key_paths)
        if private_key_paths and not self.private_key_text:
            logger.warning(
                "[GitHubApp] private key files configured but none are valid"
            )

    def _resolve_private_key_paths(self, plugin_cfg: dict[str, Any]) -> list[str]:
        candidates = _ensure_list(plugin_cfg.get("private_key_files"))

        plugin_data_root = Path(get_astrbot_plugin_data_path()) / PLUGIN_ROOT_DIR
        resolved: list[str] = []
        for candidate in candidates:
            candidate = candidate.replace("\\", "/").strip()
            if not candidate:
                continue
            if os.path.isabs(candidate):
                resolved.append(candidate)
                continue
            if candidate.startswith("files/"):
                resolved.append(str((plugin_data_root / candidate).resolve()))
            else:
                resolved.append(str((plugin_data_root / "files" / candidate).resolve()))
        return resolved

    def _build_message_event(
        self,
        parsed: GitHubParsedEvent,
        payload: Mapping,
        delivery_id: str,
    ) -> GitHubAppMessageEvent:
        abm = AstrBotMessage()
        abm.type = MessageType.GROUP_MESSAGE
        abm.self_id = self.meta().id
        abm.session_id = parsed.session_id
        abm.message_id = delivery_id or uuid.uuid4().hex
        abm.group_id = parsed.repository
        abm.sender = MessageMember(
            user_id=parsed.sender_id,
            nickname=parsed.sender_login,
        )
        abm.message = [Plain(text=parsed.summary)]
        abm.message_str = parsed.summary
        abm.raw_message = dict(payload)
        abm.timestamp = parsed.timestamp

        event = GitHubAppMessageEvent(
            message_str=abm.message_str,
            message_obj=abm,
            platform_meta=self.meta(),
            session_id=parsed.session_id,
            adapter=self,
        )
        event.set_extra("github_event", parsed.event_name)
        event.set_extra("github_action", parsed.action)
        event.set_extra("github_repository", parsed.repository)
        event.set_extra("github_session_id", parsed.session_id)
        if parsed.installation_id is not None:
            event.set_extra("github_installation_id", parsed.installation_id)
        return event
