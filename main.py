from __future__ import annotations

import asyncio
import re
import secrets
import time
from pathlib import Path
from typing import Any, Mapping

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.config.default import CONFIG_METADATA_2, WEBHOOK_SUPPORTED_PLATFORMS
from astrbot.core.skills.skill_manager import SkillManager
from astrbot.core.utils.astrbot_path import get_astrbot_skills_path
from .workflow.sandbox_workspace import (
    build_shell_workspace_bootstrap_command,
    normalize_repo_full_name,
    sanitize_workspace_session_key,
)

GITHUB_ADAPTER_TYPE = "github_app"
GITHUB_REPO_LS_TOOL_NAME = "github_repo_ls"
GITHUB_REPO_READ_TOOL_NAME = "github_repo_read"
GITHUB_REPO_SEARCH_TOOL_NAME = "github_repo_search"
DEFAULT_GITHUB_SKILL_NAME = "github_app_ops"
SUPPORTED_GITHUB_EVENTS = [
    "issues",
    "issue_comment",
    "pull_request",
    "pull_request_review",
    "pull_request_review_comment",
    "push",
    "release",
    "discussion",
    "discussion_comment",
    "watch",
    "fork",
]
RUNTIME_PLUGIN_CONFIG: dict[str, Any] = {}
IMAGE_ATTACHMENT_PATH_HINT_RE = re.compile(
    r"^\[Image Attachment:\s*path\s+.+\]$",
    re.IGNORECASE,
)
SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
GITHUB_TOKEN_LITERAL_RE = re.compile(r"\bghs_[A-Za-z0-9_]{20,}\b")
GITHUB_FAKE_TOKEN_LITERAL_RE = re.compile(r"\bghu_fake_[A-Za-z0-9]{24,}\b")
GITHUB_ONLY_LLM_TOOLS = {
    GITHUB_REPO_LS_TOOL_NAME,
    GITHUB_REPO_READ_TOOL_NAME,
    GITHUB_REPO_SEARCH_TOOL_NAME,
}


def set_runtime_plugin_config(config: dict | None) -> None:
    global RUNTIME_PLUGIN_CONFIG
    if isinstance(config, dict):
        RUNTIME_PLUGIN_CONFIG = dict(config)
    else:
        RUNTIME_PLUGIN_CONFIG = {}


def get_runtime_plugin_config() -> dict[str, Any]:
    return dict(RUNTIME_PLUGIN_CONFIG)


def _ensure_http_url_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    urls: list[str] = []
    for item in value:
        url = str(item).strip()
        if url.startswith(("http://", "https://")):
            urls.append(url)
    return list(dict.fromkeys(urls))


def _ensure_path_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    paths: list[str] = []
    for item in value:
        path = str(item).strip()
        if path:
            paths.append(path)
    return list(dict.fromkeys(paths))


async def _register_local_image_urls(local_paths: list[str]) -> list[str]:
    image_paths: list[str] = []
    for path in local_paths:
        try:
            local_path = Path(str(path).strip()).resolve()
            if local_path.is_file():
                image_paths.append(str(local_path))
        except Exception as exc:
            logger.warning(f"[GitHubApp] failed to resolve local image path: path={path}, err={exc}")
    return list(dict.fromkeys(image_paths))


def _contains_github_token_literal(text: str) -> bool:
    if not text:
        return False
    return bool(GITHUB_TOKEN_LITERAL_RE.search(text))


def _sanitize_skill_name(raw_name: Any) -> str:
    name = str(raw_name or "").strip()
    if not name:
        return DEFAULT_GITHUB_SKILL_NAME
    if SKILL_NAME_RE.fullmatch(name):
        return name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return cleaned or DEFAULT_GITHUB_SKILL_NAME


def _extract_repo_from_github_session(session_id: str) -> str:
    if not isinstance(session_id, str):
        return ""
    parts = session_id.split(":", 3)
    if len(parts) != 4:
        return ""
    if parts[0] != "github":
        return ""
    repo = parts[1].strip()
    return repo


def _extract_thread_meta_from_github_session(session_id: str) -> tuple[str, int | None]:
    if not isinstance(session_id, str):
        return "", None
    parts = session_id.split(":", 3)
    if len(parts) != 4:
        return "", None
    if parts[0] != "github":
        return "", None
    thread_type = str(parts[2]).strip()
    raw_number = str(parts[3]).strip()
    if raw_number.isdigit():
        return thread_type, int(raw_number)
    return thread_type, None


def _build_workspace_path(
    repo: str,
    github_session_id: str,
    workspace_root: str,
) -> str:
    normalized_repo = normalize_repo_full_name(repo)
    if not normalized_repo:
        return ""
    owner, name = normalized_repo.split("/", 1)
    root = str(workspace_root or "").strip() or "/tmp/github-workspaces"
    root = root.rstrip("/")
    session_key = sanitize_workspace_session_key(github_session_id)
    return f"{root}/{owner}__{name}/{session_key}"


def _build_github_skill_content(skill_name: str) -> str:
    return f"""---
description: GitHub 应用操作技能。仅使用受控工具。
---

# {skill_name}

## Priority

- 浏览仓库结构时，先调用 `{GITHUB_REPO_LS_TOOL_NAME}`。
- 阅读文件内容时，调用 `{GITHUB_REPO_READ_TOOL_NAME}`。
- 关键字检索代码时，调用 `{GITHUB_REPO_SEARCH_TOOL_NAME}`。

## Typical flow

1. 先解析仓库标识 `owner/repo`。
2. 处理代码问题时，先列目录，再分段读取文件。
"""

def _ensure_github_skill(config: Mapping[str, Any] | None) -> str:
    cfg = dict(config or {})
    if not bool(cfg.get("auto_create_github_skill", True)):
        return _sanitize_skill_name(
            cfg.get("github_skill_name", DEFAULT_GITHUB_SKILL_NAME)
        )

    skill_name = _sanitize_skill_name(
        cfg.get("github_skill_name", DEFAULT_GITHUB_SKILL_NAME)
    )
    overwrite = bool(cfg.get("overwrite_github_skill", True))
    skill_content = _build_github_skill_content(skill_name)

    try:
        skill_root = Path(get_astrbot_skills_path())
        skill_dir = skill_root / skill_name
        skill_path = skill_dir / "SKILL.md"
        skill_dir.mkdir(parents=True, exist_ok=True)

        should_write = True
        if skill_path.exists() and not overwrite:
            should_write = False
        if should_write:
            if (
                not skill_path.exists()
                or skill_path.read_text(encoding="utf-8") != skill_content
            ):
                skill_path.write_text(skill_content, encoding="utf-8")

        SkillManager().set_skill_active(skill_name, True)
        logger.info(
            f"[GitHubApp] ensured skill '{skill_name}' at {skill_path} and activated it"
        )
    except Exception as exc:
        logger.warning(f"[GitHubApp] ensure skill failed ({skill_name}): {exc}")
    return skill_name


def _inject_platform_metadata() -> None:
    if GITHUB_ADAPTER_TYPE not in WEBHOOK_SUPPORTED_PLATFORMS:
        WEBHOOK_SUPPORTED_PLATFORMS.append(GITHUB_ADAPTER_TYPE)

    platform_meta = CONFIG_METADATA_2["platform_group"]["metadata"]["platform"]
    items = platform_meta["items"]

    items["github_app_id"] = {
        "description": "平台应用编号",
        "type": "string",
        "hint": "可在平台应用设置页面中找到。",
    }
    items["github_webhook_secret"] = {
        "description": "回调密钥",
        "type": "string",
        "hint": "必须与平台应用中配置的回调密钥完全一致。",
    }
    items["github_api_base_url"] = {
        "description": "接口基础地址",
        "type": "string",
        "hint": "留空时使用平台默认接口地址。",
    }
    items["github_events"] = {
        "description": "订阅事件",
        "type": "list",
        "hint": "留空表示订阅全部已支持事件。",
        "options": SUPPORTED_GITHUB_EVENTS,
    }
    items["wake_event_types"] = {
        "description": "唤醒事件类型",
        "type": "list",
        "hint": "仅这些事件类型会按事件触发大模型唤醒。",
        "options": SUPPORTED_GITHUB_EVENTS,
    }
    items["wake_on_mentions"] = {
        "description": "@提及时唤醒",
        "type": "bool",
        "hint": "当评论正文中提及机器人时触发唤醒。",
    }
    items["mention_target_logins"] = {
        "description": "提及目标登录名",
        "type": "list",
        "hint": "仅当 @账号名 命中该列表时，按提及唤醒。",
    }
    items["ignore_bot_sender_events"] = {
        "description": "忽略机器人发送者事件",
        "type": "bool",
        "hint": "忽略发送者为平台机器人用户的事件。",
    }
    items["github_signature_validation"] = {
        "description": "启用签名校验",
        "type": "bool",
        "hint": "对每次回调请求校验签名请求头。",
    }
    items["github_delivery_cache_ttl_seconds"] = {
        "description": "投递去重缓存时长（秒）",
        "type": "int",
        "hint": "防重放窗口时长（秒）。",
    }
    items["github_delivery_cache_max_entries"] = {
        "description": "投递去重缓存最大条目数",
        "type": "int",
        "hint": "内存去重缓存上限。",
    }

    logger.info("[GitHubApp] platform metadata injected")


@register(
    "astrbot_plugin_githubapp-adapter",
    "OpenCode",
    "为 AstrBot 提供 GitHub App 回调适配与受控仓库操作能力。",
    "v0.2.0",
    "https://github.com/example/astrbot_plugin_githubapp-adapter",
)
class GitHubAppAdapterPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self._fake_token_lock = asyncio.Lock()
        self._session_fake_tokens: dict[str, dict[str, Any]] = {}
        self._fake_token_to_session: dict[str, str] = {}
        skill_name = _ensure_github_skill(self.config)
        runtime_cfg = dict(self.config)
        runtime_cfg["effective_github_skill_name"] = skill_name
        set_runtime_plugin_config(runtime_cfg)
        _inject_platform_metadata()
        from .adapter.github_app_adapter import GitHubAppAdapter  # noqa: F401

    def _resolve_github_adapter(
        self,
        event: AstrMessageEvent,
        platform_id: str = "",
    ) -> Any | None:
        candidate: Any | None = None
        target_platform_id = str(platform_id or "").strip()
        if target_platform_id:
            candidate = self.context.get_platform_inst(target_platform_id)
        if candidate is None and event.get_platform_name() == GITHUB_ADAPTER_TYPE:
            candidate = self.context.get_platform_inst(event.get_platform_id())
        if candidate is None:
            candidate = self.context.get_platform(GITHUB_ADAPTER_TYPE)
        if candidate is None:
            return None
        try:
            if candidate.meta().name != GITHUB_ADAPTER_TYPE:
                return None
        except Exception:
            return None
        return candidate

    def _resolve_repo_from_event(self, event: AstrMessageEvent, repo: str) -> str:
        repo_value = str(repo or "").strip()
        if not repo_value:
            repo_value = str(event.get_extra("github_repository", "")).strip()
        if not repo_value:
            repo_value = _extract_repo_from_github_session(
                str(event.get_extra("github_session_id", "")).strip()
            )
        if not repo_value and event.get_platform_name() == GITHUB_ADAPTER_TYPE:
            repo_value = _extract_repo_from_github_session(event.get_session_id())
        return repo_value

    def _resolve_github_session_id(self, event: AstrMessageEvent) -> str:
        github_session_id = str(event.get_extra("github_session_id", "")).strip()
        if not github_session_id and event.get_platform_name() == GITHUB_ADAPTER_TYPE:
            github_session_id = event.get_session_id()
        return github_session_id

    def _resolve_installation_id_from_event(self, event: AstrMessageEvent) -> int | None:
        raw = event.get_extra("github_installation_id", 0)
        try:
            resolved = int(raw or 0)
        except Exception:
            return None
        return resolved if resolved > 0 else None

    def _cleanup_fake_token_cache(self, now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        expired_sessions: list[str] = []
        for session_id, entry in self._session_fake_tokens.items():
            expires_at = float(entry.get("expires_at_epoch", 0.0) or 0.0)
            if expires_at <= ts:
                expired_sessions.append(session_id)
        for session_id in expired_sessions:
            entry = self._session_fake_tokens.pop(session_id, {})
            fake_token = str(entry.get("fake_token", "")).strip()
            if fake_token and self._fake_token_to_session.get(fake_token) == session_id:
                self._fake_token_to_session.pop(fake_token, None)

    @staticmethod
    def _build_fake_token_literal() -> str:
        return f"ghu_fake_{secrets.token_hex(18)}"

    async def _ensure_session_fake_token_bridge(
        self,
        event: AstrMessageEvent,
        adapter: Any,
        repo: str,
        cfg: Mapping[str, Any],
    ) -> tuple[str, int]:
        if not bool(cfg.get("enable_fake_token_bridge", True)):
            return "", 0
        if not hasattr(adapter, "issue_readonly_token_for_skill"):
            return "", 0

        normalized_repo = normalize_repo_full_name(repo)
        if not normalized_repo:
            return "", 0
        session_id = self._resolve_github_session_id(event)
        if not session_id:
            return "", 0

        try:
            fake_ttl_seconds = int(cfg.get("fake_token_ttl_seconds", 900))
        except Exception:
            fake_ttl_seconds = 900
        fake_ttl_seconds = min(max(60, fake_ttl_seconds), 3600)
        now = time.time()

        async with self._fake_token_lock:
            self._cleanup_fake_token_cache(now)
            cached = self._session_fake_tokens.get(session_id)
            if isinstance(cached, dict):
                cached_repo = str(cached.get("repo", "")).strip()
                cached_token = str(cached.get("fake_token", "")).strip()
                cached_expire = float(cached.get("expires_at_epoch", 0.0) or 0.0)
                if (
                    cached_repo == normalized_repo
                    and cached_token
                    and cached_expire - 20 > now
                ):
                    return cached_token, int(max(1, cached_expire - now))

        installation_id = self._resolve_installation_id_from_event(event)
        ok, payload = await adapter.issue_readonly_token_for_skill(
            repo=normalized_repo,
            installation_id=installation_id,
            max_ttl_seconds=fake_ttl_seconds,
        )
        if not ok:
            logger.warning(
                "[GitHubApp] issue readonly token for bridge failed: "
                f"repo={normalized_repo}, detail={payload}"
            )
            return "", 0

        real_token = str(payload.get("token", "")).strip()
        if not real_token:
            return "", 0
        try:
            real_expires = float(payload.get("expires_at_epoch", 0.0) or 0.0)
        except Exception:
            real_expires = 0.0
        if real_expires <= now:
            real_expires = now + fake_ttl_seconds
        effective_expire = min(real_expires, now + fake_ttl_seconds)
        fake_token = self._build_fake_token_literal()

        async with self._fake_token_lock:
            self._cleanup_fake_token_cache(now)
            old_entry = self._session_fake_tokens.get(session_id, {})
            old_fake = str(old_entry.get("fake_token", "")).strip()
            if old_fake:
                self._fake_token_to_session.pop(old_fake, None)
            self._session_fake_tokens[session_id] = {
                "repo": normalized_repo,
                "installation_id": int(payload.get("installation_id", 0) or 0),
                "fake_token": fake_token,
                "real_token": real_token,
                "expires_at_epoch": effective_expire,
            }
            self._fake_token_to_session[fake_token] = session_id
        return fake_token, int(max(1, effective_expire - now))

    async def _replace_fake_token_literal_for_tool(
        self,
        event: AstrMessageEvent,
        text: str,
    ) -> tuple[str, bool, str]:
        if not text:
            return text, False, ""
        session_id = self._resolve_github_session_id(event)
        if not session_id:
            return text, False, ""
        now = time.time()
        async with self._fake_token_lock:
            self._cleanup_fake_token_cache(now)
            entry = self._session_fake_tokens.get(session_id)
            if not isinstance(entry, dict):
                if GITHUB_FAKE_TOKEN_LITERAL_RE.search(text):
                    return text, False, "fake_token_missing"
                return text, False, ""
            fake_token = str(entry.get("fake_token", "")).strip()
            real_token = str(entry.get("real_token", "")).strip()
            expires_at = float(entry.get("expires_at_epoch", 0.0) or 0.0)

        if fake_token and fake_token in text:
            if expires_at <= now:
                return text, False, "fake_token_expired"
            if real_token:
                return text.replace(fake_token, real_token), True, ""
            return text, False, "fake_token_missing"
        if GITHUB_FAKE_TOKEN_LITERAL_RE.search(text):
            return text, False, "fake_token_missing"
        return text, False, ""

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        skill_name = _ensure_github_skill(self.config)
        runtime_cfg = dict(self.config)
        runtime_cfg["effective_github_skill_name"] = skill_name
        set_runtime_plugin_config(runtime_cfg)
        _inject_platform_metadata()

    @filter.on_using_llm_tool(priority=-20000)
    async def guard_github_tool_usage(
        self,
        event: AstrMessageEvent,
        tool: Any,
        tool_args: dict | None,
    ) -> None:
        if event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            return
        if not isinstance(tool_args, dict):
            return

        cfg = get_runtime_plugin_config()
        tool_name = str(getattr(tool, "name", "")).strip()
        if tool_name not in {"astrbot_execute_shell", "astrbot_execute_python"}:
            return

        source_key = "command" if tool_name == "astrbot_execute_shell" else "code"
        source_text = str(tool_args.get(source_key, ""))
        runtime_text = source_text

        if tool_name == "astrbot_execute_shell":
            auto_prepare_workspace = bool(
                cfg.get("enable_auto_sandbox_workspace_prepare", True)
            )
            if auto_prepare_workspace:
                repo_value = str(event.get_extra("github_repository", "")).strip()
                if not repo_value:
                    repo_value = _extract_repo_from_github_session(
                        str(event.get_extra("github_session_id", "")).strip()
                    )
                if not repo_value and event.get_platform_name() == GITHUB_ADAPTER_TYPE:
                    repo_value = _extract_repo_from_github_session(event.get_session_id())

                normalized_repo = normalize_repo_full_name(repo_value)
                if normalized_repo:
                    workspace_root = str(
                        cfg.get("sandbox_workspace_root", "/tmp/github-workspaces")
                    ).strip() or "/tmp/github-workspaces"
                    try:
                        clone_depth = int(cfg.get("sandbox_workspace_clone_depth", 1))
                    except Exception:
                        clone_depth = 1
                    github_session_id = self._resolve_github_session_id(event)
                    session_key = sanitize_workspace_session_key(github_session_id)
                    runtime_text = build_shell_workspace_bootstrap_command(
                        command=source_text,
                        repo=normalized_repo,
                        session_key=session_key,
                        workspace_root=workspace_root,
                        clone_depth=clone_depth,
                    )

        enforce_guard = bool(cfg.get("enforce_tool_write_guard", False))
        block_token_literal = (
            bool(cfg.get("guard_block_token_literal", True)) if enforce_guard else False
        )

        reasons: list[str] = []
        if enforce_guard and block_token_literal:
            if tool_name == "astrbot_execute_shell":
                if _contains_github_token_literal(source_text):
                    reasons.append("token_literal_in_shell_command")
            elif tool_name == "astrbot_execute_python":
                if _contains_github_token_literal(source_text):
                    reasons.append("token_literal_in_python_code")

        if reasons:
            logger.warning(
                f"[GitHubApp] blocked risky tool call: tool={tool_name}, reasons={reasons}"
            )
            reason_text = ",".join(list(dict.fromkeys(reasons)))
            message = f"BLOCKED by github_app guard: {reason_text}"
            if tool_name == "astrbot_execute_shell":
                safe_message = message.replace('"', "'").replace("\n", " ")
                tool_args["command"] = f'echo "{safe_message}"'
            else:
                tool_args["code"] = f"print({message!r})"
            return

        enable_fake_token_bridge = bool(cfg.get("enable_fake_token_bridge", True))
        if enable_fake_token_bridge:
            replaced_text, replaced, replace_reason = await self._replace_fake_token_literal_for_tool(
                event,
                runtime_text,
            )
            if replace_reason:
                safe_reason = replace_reason.replace('"', "'")
                message = f"BLOCKED by github_app guard: {safe_reason}"
                if tool_name == "astrbot_execute_shell":
                    tool_args["command"] = f'echo "{message}"'
                else:
                    tool_args["code"] = f"print({message!r})"
                logger.warning(
                    "[GitHubApp] blocked tool call because fake token cannot be resolved: "
                    f"tool={tool_name}, reason={replace_reason}"
                )
                return
            runtime_text = replaced_text
            if replaced:
                logger.info(
                    "[GitHubApp] replaced fake token literal before tool execution: "
                    f"tool={tool_name}"
                )

        tool_args[source_key] = runtime_text

    @filter.llm_tool(name=GITHUB_REPO_LS_TOOL_NAME)
    async def github_repo_ls(
        self,
        event: AstrMessageEvent,
        repo: str = "",
        path: str = ".",
        ref: str = "",
        offset: int = 0,
        limit: int = 50,
        platform_id: str = "",
    ) -> str:
        """列出仓库目录的一层文件列表（适合先浏览再深入）。

        Args:
            repo(string): 目标仓库，格式 owner/repo；为空时从当前 GitHub 会话自动解析。
            path(string): 目录路径，默认 "." 表示仓库根目录。
            ref(string): 可选，分支/标签/提交；为空时使用默认分支。
            offset(number): 可选，分页偏移，默认 0。
            limit(number): 可选，分页大小，默认 50，最大 200。
            platform_id(string): 可选，当存在多个 github_app 平台时可指定。
        """
        if event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            return "该工具仅在 github_app 平台会话中可用。"

        adapter = self._resolve_github_adapter(event, platform_id)
        if adapter is None:
            return "未找到可用的 github_app 平台适配器。"
        if not hasattr(adapter, "list_repo_dir_for_skill"):
            return "当前 github_app 适配器不支持目录浏览工具，请升级插件。"

        repo_value = self._resolve_repo_from_event(event, repo)
        if not repo_value:
            return "无法解析 repo，请显式传入 owner/repo。"

        path_value = str(path or "").strip() or "."
        ref_value = str(ref or "").strip()
        try:
            offset_value = max(0, int(offset or 0))
        except Exception:
            offset_value = 0
        try:
            limit_value = int(limit or 50)
        except Exception:
            limit_value = 50
        limit_value = min(max(1, limit_value), 200)

        ok, payload = await adapter.list_repo_dir_for_skill(
            repo=repo_value,
            path=path_value,
            ref=ref_value,
            offset=offset_value,
            limit=limit_value,
        )
        if not ok:
            detail = str(payload.get("error", "unknown error"))
            stage = str(payload.get("stage", "")).strip()
            if stage:
                return f"读取仓库目录失败（{stage}）：{detail}"
            return f"读取仓库目录失败：{detail}"

        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        resolved_path = str(payload.get("path", path_value)).strip()
        resolved_ref = str(payload.get("ref", ref_value)).strip()
        total = int(payload.get("total", len(entries)) or 0)
        current_offset = int(payload.get("offset", offset_value) or 0)
        current_limit = int(payload.get("limit", limit_value) or 0)
        has_more = bool(payload.get("has_more", False))

        lines = [
            "仓库目录读取成功。",
            f"repo: {repo_value}",
            f"path: {resolved_path}",
            f"ref: {resolved_ref or '(default)'}",
            f"offset: {current_offset}",
            f"limit: {current_limit}",
            f"total: {total}",
            "entries:",
        ]
        for idx, item in enumerate(entries, start=current_offset + 1):
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type", "")).strip().lower()
            item_name = str(item.get("name", "")).strip()
            item_path = str(item.get("path", "")).strip()
            item_size = int(item.get("size", 0) or 0)
            mark = "DIR" if item_type == "dir" else "FILE"
            lines.append(f"{idx}. [{mark}] {item_name} ({item_size} bytes) path={item_path}")
        if has_more:
            lines.append("note: 结果未展示完，继续调用 github_repo_ls 并增大 offset。")
        return "\n".join(lines)

    @filter.llm_tool(name=GITHUB_REPO_READ_TOOL_NAME)
    async def github_repo_read(
        self,
        event: AstrMessageEvent,
        repo: str = "",
        path: str = "",
        ref: str = "",
        start_line: int = 1,
        max_lines: int = 200,
        platform_id: str = "",
    ) -> str:
        """按行读取仓库文件内容（大文件建议分段读取）。

        Args:
            repo(string): 目标仓库，格式 owner/repo；为空时从当前 GitHub 会话自动解析。
            path(string): 文件路径，例如 README.md 或 src/main.py。
            ref(string): 可选，分支/标签/提交；为空时使用默认分支。
            start_line(number): 起始行号（1-based），默认 1。
            max_lines(number): 读取行数，默认 200，最大 400。
            platform_id(string): 可选，当存在多个 github_app 平台时可指定。
        """
        if event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            return "该工具仅在 github_app 平台会话中可用。"

        adapter = self._resolve_github_adapter(event, platform_id)
        if adapter is None:
            return "未找到可用的 github_app 平台适配器。"
        if not hasattr(adapter, "read_repo_file_for_skill"):
            return "当前 github_app 适配器不支持文件读取工具，请升级插件。"

        repo_value = self._resolve_repo_from_event(event, repo)
        if not repo_value:
            return "无法解析 repo，请显式传入 owner/repo。"

        path_value = str(path or "").strip()
        if not path_value:
            return "缺少 path 参数，请传入文件路径。"
        ref_value = str(ref or "").strip()
        try:
            start_value = max(1, int(start_line or 1))
        except Exception:
            start_value = 1
        try:
            max_lines_value = int(max_lines or 200)
        except Exception:
            max_lines_value = 200
        max_lines_value = min(max(1, max_lines_value), 400)

        ok, payload = await adapter.read_repo_file_for_skill(
            repo=repo_value,
            path=path_value,
            ref=ref_value,
            start_line=start_value,
            max_lines=max_lines_value,
        )
        if not ok:
            detail = str(payload.get("error", "unknown error"))
            stage = str(payload.get("stage", "")).strip()
            if stage:
                return f"读取仓库文件失败（{stage}）：{detail}"
            return f"读取仓库文件失败：{detail}"

        resolved_path = str(payload.get("path", path_value)).strip()
        resolved_ref = str(payload.get("ref", ref_value)).strip()
        sha = str(payload.get("sha", "")).strip()
        size = int(payload.get("size", 0) or 0)
        line_start = int(payload.get("line_start", start_value) or start_value)
        line_end = int(payload.get("line_end", line_start) or line_start)
        total_lines = int(payload.get("total_lines", 0) or 0)
        has_more = bool(payload.get("has_more", False))
        next_start_line = int(payload.get("next_start_line", line_end + 1) or (line_end + 1))
        content = str(payload.get("content", ""))

        lines = [
            "仓库文件读取成功。",
            f"repo: {repo_value}",
            f"path: {resolved_path}",
            f"ref: {resolved_ref or '(default)'}",
            f"sha: {sha}",
            f"size: {size}",
            f"line_range: {line_start}-{line_end}",
            f"total_lines: {total_lines}",
            f"has_more: {str(has_more).lower()}",
        ]
        if has_more:
            lines.append(f"next_start_line: {next_start_line}")
        lines.append("content:")
        lines.append(content)
        return "\n".join(lines)

    @filter.llm_tool(name=GITHUB_REPO_SEARCH_TOOL_NAME)
    async def github_repo_search(
        self,
        event: AstrMessageEvent,
        repo: str = "",
        query: str = "",
        path: str = "",
        limit: int = 20,
        platform_id: str = "",
    ) -> str:
        """在仓库内按关键字搜索代码路径与片段。

        Args:
            repo(string): 目标仓库，格式 owner/repo；为空时从当前 GitHub 会话自动解析。
            query(string): 搜索关键词。
            path(string): 可选，限制在某个目录路径下搜索。
            limit(number): 可选，返回条数，默认 20，最大 50。
            platform_id(string): 可选，当存在多个 github_app 平台时可指定。
        """
        if event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            return "该工具仅在 github_app 平台会话中可用。"

        adapter = self._resolve_github_adapter(event, platform_id)
        if adapter is None:
            return "未找到可用的 github_app 平台适配器。"
        if not hasattr(adapter, "search_repo_code_for_skill"):
            return "当前 github_app 适配器不支持仓库搜索工具，请升级插件。"

        repo_value = self._resolve_repo_from_event(event, repo)
        if not repo_value:
            return "无法解析 repo，请显式传入 owner/repo。"

        query_value = str(query or "").strip()
        if not query_value:
            return "缺少 query 参数，请提供搜索关键词。"
        path_value = str(path or "").strip()
        try:
            limit_value = int(limit or 20)
        except Exception:
            limit_value = 20
        limit_value = min(max(1, limit_value), 50)

        ok, payload = await adapter.search_repo_code_for_skill(
            repo=repo_value,
            query=query_value,
            path=path_value,
            limit=limit_value,
        )
        if not ok:
            detail = str(payload.get("error", "unknown error"))
            stage = str(payload.get("stage", "")).strip()
            if stage:
                return f"仓库搜索失败（{stage}）：{detail}"
            return f"仓库搜索失败：{detail}"

        hits = payload.get("hits", [])
        if not isinstance(hits, list):
            hits = []
        total_count = int(payload.get("total_count", len(hits)) or 0)
        incomplete = bool(payload.get("incomplete_results", False))
        lines = [
            "仓库代码搜索成功。",
            f"repo: {repo_value}",
            f"query: {query_value}",
            f"path_filter: {path_value or '(none)'}",
            f"total_count: {total_count}",
            f"returned: {len(hits)}",
            f"incomplete_results: {str(incomplete).lower()}",
            "hits:",
        ]
        for idx, hit in enumerate(hits, start=1):
            if not isinstance(hit, Mapping):
                continue
            hit_path = str(hit.get("path", "")).strip()
            score = float(hit.get("score", 0.0) or 0.0)
            snippet = str(hit.get("snippet", "")).strip()
            lines.append(f"{idx}. {hit_path} (score={score:.2f})")
            if snippet:
                lines.append(f"   snippet: {snippet}")
        return "\n".join(lines)

    @filter.on_llm_request(priority=-20000)
    async def fix_github_image_llm_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        func_tool = req.func_tool
        if func_tool is not None and event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            for tool_name in GITHUB_ONLY_LLM_TOOLS:
                try:
                    func_tool.remove_tool(tool_name)
                except Exception:
                    pass

        if event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            return

        cfg = get_runtime_plugin_config()
        repo_value = str(event.get_extra("github_repository", "")).strip()
        github_session_id = str(event.get_extra("github_session_id", "")).strip()
        if not github_session_id:
            github_session_id = event.get_session_id()
        if not repo_value:
            repo_value = _extract_repo_from_github_session(github_session_id)

        thread_type = str(event.get_extra("github_thread_type", "")).strip()
        thread_number_raw = str(event.get_extra("github_thread_number", "")).strip()
        thread_number = int(thread_number_raw) if thread_number_raw.isdigit() else None
        if not thread_type:
            parsed_type, parsed_num = _extract_thread_meta_from_github_session(
                github_session_id
            )
            thread_type = parsed_type
            if thread_number is None:
                thread_number = parsed_num

        thread_title = str(event.get_extra("github_thread_title", "")).strip()
        thread_url = str(event.get_extra("github_thread_url", "")).strip()
        workspace_hint = ""
        if bool(cfg.get("enable_auto_sandbox_workspace_prepare", True)) and repo_value:
            workspace_root = str(
                cfg.get("sandbox_workspace_root", "/tmp/github-workspaces")
            ).strip() or "/tmp/github-workspaces"
            workspace_hint = _build_workspace_path(
                repo=repo_value,
                github_session_id=github_session_id,
                workspace_root=workspace_root,
            )
        fake_token_literal = ""
        fake_token_ttl_left = 0
        normalized_repo = normalize_repo_full_name(repo_value)
        if bool(cfg.get("enable_fake_token_bridge", True)) and normalized_repo:
            adapter = self._resolve_github_adapter(event)
            if adapter is not None:
                fake_token_literal, fake_token_ttl_left = await self._ensure_session_fake_token_bridge(
                    event=event,
                    adapter=adapter,
                    repo=normalized_repo,
                    cfg=cfg,
                )

        context_lines = [
            "[GitHub 会话上下文]",
            f"- 仓库: {repo_value or '未知'}",
            f"- 会话编号: {github_session_id or '未知'}",
        ]
        if thread_type:
            context_lines.append(f"- 线程类型: {thread_type}")
        if thread_number is not None:
            context_lines.append(f"- 线程编号: {thread_number}")
        if thread_title:
            context_lines.append(f"- 线程标题: {thread_title}")
        if thread_url:
            context_lines.append(f"- 线程链接: {thread_url}")
        if workspace_hint:
            context_lines.append(f"- 沙盒工作区: {workspace_hint}")
            context_lines.append(
                "- 说明: shell 工具会先进入该工作区，缺少仓库时自动克隆。"
            )
        if fake_token_literal:
            context_lines.append(f"- 只读临时令牌占位符: {fake_token_literal}")
            context_lines.append(f"- 占位符有效期（秒）: {fake_token_ttl_left}")
            context_lines.append(
                "- 说明: 工具执行前会自动替换为真实令牌；只用于当前仓库只读访问。"
            )
        context_lines.append(
            "- 指令: 遇到仓库代码问题，先调用 github_repo_ls，再用 github_repo_read 或 github_repo_search。"
        )
        context_block = "\n".join(context_lines).strip()
        if context_block and context_block not in req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n{context_block}".strip()

        local_paths = _ensure_path_list(event.get_extra("github_image_local_paths", []))
        failed_urls = _ensure_http_url_list(event.get_extra("github_image_failed_urls", []))
        origin_urls = _ensure_http_url_list(event.get_extra("github_image_urls", []))
        existing_urls = list(dict.fromkeys(str(u).strip() for u in (req.image_urls or [])))
        existing_urls = [u for u in existing_urls if u.startswith(("http://", "https://"))]

        if local_paths:
            local_image_paths = await _register_local_image_urls(local_paths)
            if local_image_paths:
                # Use local paths to avoid one-time file-token expiration warnings.
                req.image_urls = local_image_paths
            elif existing_urls:
                req.image_urls = existing_urls
            elif failed_urls:
                req.image_urls = failed_urls
            elif origin_urls:
                req.image_urls = origin_urls
        elif not existing_urls:
            if failed_urls:
                req.image_urls = failed_urls
            elif origin_urls:
                req.image_urls = origin_urls

        removed_hint_parts = 0
        kept_parts = []
        for part in req.extra_user_content_parts:
            part_type = str(getattr(part, "type", "")).lower()
            if part_type == "text":
                text = str(getattr(part, "text", "")).strip()
                if IMAGE_ATTACHMENT_PATH_HINT_RE.match(text):
                    removed_hint_parts += 1
                    continue
            kept_parts.append(part)
        if removed_hint_parts:
            req.extra_user_content_parts = kept_parts

        if not req.image_urls:
            return

        hint = (
            "图片已通过多模态输入附带。"
            "不要调用工具读取本地图片路径，请直接基于已附带图片进行分析。"
        )
        if hint not in req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n{hint}".strip()


# 兼容旧类名，避免外部导入或旧配置失效
GitHubAppAdopterPlugin = GitHubAppAdapterPlugin
