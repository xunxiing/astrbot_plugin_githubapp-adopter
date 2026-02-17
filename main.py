from __future__ import annotations

import re
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.config.default import CONFIG_METADATA_2, WEBHOOK_SUPPORTED_PLATFORMS

GITHUB_ADAPTER_TYPE = "github_app"
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


def set_runtime_plugin_config(config: dict | None) -> None:
    global RUNTIME_PLUGIN_CONFIG
    if isinstance(config, dict):
        RUNTIME_PLUGIN_CONFIG = dict(config)
    else:
        RUNTIME_PLUGIN_CONFIG = {}


def get_runtime_plugin_config() -> dict[str, Any]:
    return dict(RUNTIME_PLUGIN_CONFIG)


def _inject_platform_metadata() -> None:
    if GITHUB_ADAPTER_TYPE not in WEBHOOK_SUPPORTED_PLATFORMS:
        WEBHOOK_SUPPORTED_PLATFORMS.append(GITHUB_ADAPTER_TYPE)

    platform_meta = CONFIG_METADATA_2["platform_group"]["metadata"]["platform"]
    items = platform_meta["items"]

    items.setdefault(
        "github_app_id",
        {
            "description": "GitHub App ID",
            "type": "string",
            "hint": "GitHub App 设置页中的 App ID。",
        },
    )
    items.setdefault(
        "github_webhook_secret",
        {
            "description": "GitHub Webhook 密钥",
            "type": "string",
            "hint": "必须与 GitHub App 中配置的 Webhook Secret 保持一致。",
        },
    )
    items.setdefault(
        "github_api_base_url",
        {
            "description": "GitHub API 基础地址",
            "type": "string",
            "hint": "默认值为 https://api.github.com。",
        },
    )
    items.setdefault(
        "github_events",
        {
            "description": "GitHub 事件订阅",
            "type": "list",
            "hint": "留空表示订阅全部已支持事件。",
            "options": SUPPORTED_GITHUB_EVENTS,
        },
    )
    items.setdefault(
        "wake_event_types",
        {
            "description": "唤醒事件类型",
            "type": "list",
            "hint": "仅这些事件会触发 LLM 唤醒。",
            "options": SUPPORTED_GITHUB_EVENTS,
        },
    )
    items.setdefault(
        "wake_on_mentions",
        {
            "description": "Wake on @mention",
            "type": "bool",
            "hint": "Wake LLM when message body contains a GitHub @mention.",
        },
    )
    items.setdefault(
        "mention_target_logins",
        {
            "description": "Mention target logins",
            "type": "list",
            "hint": "Only wake by mention when @login matches one of these values.",
        },
    )
    items.setdefault(
        "ignore_bot_sender_events",
        {
            "description": "Ignore bot sender events",
            "type": "bool",
            "hint": "Ignore webhook events when sender is a GitHub Bot account.",
        },
    )
    items.setdefault(
        "github_signature_validation",
        {
            "description": "启用签名校验",
            "type": "bool",
            "hint": "校验每次 Webhook 请求的 X-Hub-Signature-256。",
        },
    )
    items.setdefault(
        "github_delivery_cache_ttl_seconds",
        {
            "description": "Delivery 去重 TTL（秒）",
            "type": "int",
            "hint": "重放保护窗口。",
        },
    )
    items.setdefault(
        "github_delivery_cache_max_entries",
        {
            "description": "Delivery 去重最大条目数",
            "type": "int",
            "hint": "内存去重缓存容量上限。",
        },
    )

    logger.info("[GitHubApp] platform metadata injected")


@register(
    "astrbot_plugin_githubapp-adopter",
    "OpenCode",
    "为 AstrBot 提供 GitHub App Webhook 平台适配器",
    "v0.1.2",
    "https://github.com/example/astrbot_plugin_githubapp-adopter",
)
class GitHubAppAdopterPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        set_runtime_plugin_config(config)
        _inject_platform_metadata()
        from .adapter.github_app_adapter import GitHubAppAdapter  # noqa: F401

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        set_runtime_plugin_config(self.config)
        _inject_platform_metadata()

    @filter.on_llm_request(priority=-20000)
    async def fix_github_image_llm_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ):
        if event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            return
        if not req.image_urls:
            return

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

        hint = (
            "The image is already provided via multimodal input. "
            "Do not call tools just to open local file paths; "
            "analyze the attached image directly."
        )
        if hint not in req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n{hint}".strip()
