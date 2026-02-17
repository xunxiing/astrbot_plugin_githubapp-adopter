from __future__ import annotations

import copy
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
METADATA_I18N_DISABLED = False


def _disable_config_metadata_i18n() -> None:
    global METADATA_I18N_DISABLED
    if METADATA_I18N_DISABLED:
        return
    try:
        from astrbot.core.config.i18n_utils import ConfigMetadataI18n

        def _pass_through_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
            return copy.deepcopy(metadata)

        ConfigMetadataI18n.convert_to_i18n_keys = staticmethod(_pass_through_metadata)
        METADATA_I18N_DISABLED = True
        logger.info("[GitHubApp] 已关闭配置元数据 i18n key 转换")
    except Exception as exc:
        logger.warning(f"[GitHubApp] 关闭配置元数据 i18n 转换失败: {exc}")


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

    items["github_app_id"] = {
        "description": "GitHub App ID",
        "type": "string",
        "hint": "在 GitHub App 设置页中可找到 App ID。",
    }
    items["github_webhook_secret"] = {
        "description": "GitHub Webhook 密钥",
        "type": "string",
        "hint": "需与 GitHub App 中配置的 Webhook Secret 完全一致。",
    }
    items["github_api_base_url"] = {
        "description": "GitHub API 基础地址",
        "type": "string",
        "hint": "默认值为 https://api.github.com 。",
    }
    items["github_events"] = {
        "description": "GitHub 订阅事件",
        "type": "list",
        "hint": "留空表示订阅所有已支持事件。",
        "options": SUPPORTED_GITHUB_EVENTS,
    }
    items["wake_event_types"] = {
        "description": "唤醒事件类型",
        "type": "list",
        "hint": "仅这些事件会触发 LLM 唤醒。",
        "options": SUPPORTED_GITHUB_EVENTS,
    }
    items["wake_on_mentions"] = {
        "description": "@提及时唤醒",
        "type": "bool",
        "hint": "消息正文出现 GitHub @提及时触发唤醒。",
    }
    items["mention_target_logins"] = {
        "description": "提及目标登录名",
        "type": "list",
        "hint": "仅当 @login 命中该列表时，按 @提及唤醒。",
    }
    items["ignore_bot_sender_events"] = {
        "description": "忽略 Bot 发送者事件",
        "type": "bool",
        "hint": "当事件发送者是 GitHub Bot 账号时忽略该事件。",
    }
    items["github_signature_validation"] = {
        "description": "启用签名校验",
        "type": "bool",
        "hint": "校验每次 Webhook 请求的 X-Hub-Signature-256。",
    }
    items["github_delivery_cache_ttl_seconds"] = {
        "description": "Delivery 去重 TTL（秒）",
        "type": "int",
        "hint": "重放保护窗口（秒）。",
    }
    items["github_delivery_cache_max_entries"] = {
        "description": "Delivery 去重最大条目数",
        "type": "int",
        "hint": "内存去重缓存容量上限。",
    }

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
        _disable_config_metadata_i18n()
        set_runtime_plugin_config(config)
        _inject_platform_metadata()
        from .adapter.github_app_adapter import GitHubAppAdapter  # noqa: F401

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self):
        _disable_config_metadata_i18n()
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
            "图片已通过多模态输入提供。"
            "不要为了读取本地路径而调用工具，直接基于已附带图片进行分析。"
        )
        if hint not in req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n{hint}".strip()
