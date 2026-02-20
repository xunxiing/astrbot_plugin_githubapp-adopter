from __future__ import annotations

import re
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
GITHUB_CREATE_LICENSE_PR_TOOL_NAME = "github_create_license_pr"
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
GITHUB_ONLY_LLM_TOOLS = {
    GITHUB_CREATE_LICENSE_PR_TOOL_NAME,
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


def _extract_issue_number_from_github_session(session_id: str) -> int:
    if not isinstance(session_id, str):
        return 0
    parts = session_id.split(":", 3)
    if len(parts) != 4:
        return 0
    if parts[0] != "github" or parts[2] != "issue":
        return 0
    raw_number = str(parts[3]).strip()
    if not raw_number.isdigit():
        return 0
    return int(raw_number)


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
description: GitHub App operations skill. Use safe tools only.
---

# {skill_name}

## Priority

- For add-license tasks, call `{GITHUB_CREATE_LICENSE_PR_TOOL_NAME}`.
- For repository navigation, call `{GITHUB_REPO_LS_TOOL_NAME}` first.
- For file content, call `{GITHUB_REPO_READ_TOOL_NAME}`.
- For keyword lookup, call `{GITHUB_REPO_SEARCH_TOOL_NAME}`.

## Typical flow

1. Resolve repo as `owner/repo`.
2. If task is add-license-and-open-pr, use license PR tool.
3. For code questions, list directory first, then read files in chunks.
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
        "description": "GitHub App ID",
        "type": "string",
        "hint": "可在 GitHub App 设置页面中找到。",
    }
    items["github_webhook_secret"] = {
        "description": "GitHub Webhook 密钥",
        "type": "string",
        "hint": "必须与 GitHub App 中配置的 Webhook Secret 完全一致。",
    }
    items["github_api_base_url"] = {
        "description": "GitHub API 基础地址",
        "type": "string",
        "hint": "默认值为 https://api.github.com。",
    }
    items["github_events"] = {
        "description": "GitHub 订阅事件",
        "type": "list",
        "hint": "留空表示订阅全部已支持事件。",
        "options": SUPPORTED_GITHUB_EVENTS,
    }
    items["wake_event_types"] = {
        "description": "唤醒事件类型",
        "type": "list",
        "hint": "仅这些事件类型会按事件触发 LLM 唤醒。",
        "options": SUPPORTED_GITHUB_EVENTS,
    }
    items["wake_on_mentions"] = {
        "description": "@提及时唤醒",
        "type": "bool",
        "hint": "当 GitHub 评论正文中提及机器人时触发唤醒。",
    }
    items["mention_target_logins"] = {
        "description": "提及目标登录名",
        "type": "list",
        "hint": "仅当 @login 命中该列表时，按提及唤醒。",
    }
    items["ignore_bot_sender_events"] = {
        "description": "忽略 Bot 发送者事件",
        "type": "bool",
        "hint": "忽略 sender 为 GitHub Bot 用户的事件。",
    }
    items["github_signature_validation"] = {
        "description": "启用签名校验",
        "type": "bool",
        "hint": "对每次 webhook 请求校验 X-Hub-Signature-256。",
    }
    items["github_delivery_cache_ttl_seconds"] = {
        "description": "Delivery 去重 TTL（秒）",
        "type": "int",
        "hint": "防重放窗口时长（秒）。",
    }
    items["github_delivery_cache_max_entries"] = {
        "description": "Delivery 去重最大条目数",
        "type": "int",
        "hint": "内存去重缓存上限。",
    }

    logger.info("[GitHubApp] platform metadata injected")


@register(
    "astrbot_plugin_githubapp-adopter",
    "OpenCode",
    "为 AstrBot 提供 GitHub App Webhook 适配与临时令牌能力。",
    "v0.2.0",
    "https://github.com/example/astrbot_plugin_githubapp-adopter",
)
class GitHubAppAdopterPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
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

        if tool_name == "astrbot_execute_shell":
            raw_command = str(tool_args.get("command", ""))
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
                    github_session_id = str(event.get_extra("github_session_id", "")).strip()
                    if not github_session_id and event.get_platform_name() == GITHUB_ADAPTER_TYPE:
                        github_session_id = event.get_session_id()
                    session_key = sanitize_workspace_session_key(github_session_id)
                    tool_args["command"] = build_shell_workspace_bootstrap_command(
                        command=raw_command,
                        repo=normalized_repo,
                        session_key=session_key,
                        workspace_root=workspace_root,
                        clone_depth=clone_depth,
                    )

        enforce_guard = bool(cfg.get("enforce_tool_write_guard", False))
        if not enforce_guard:
            return

        block_token_literal = bool(cfg.get("guard_block_token_literal", True))
        if not block_token_literal:
            return

        reasons: list[str] = []
        if tool_name == "astrbot_execute_shell":
            command = str(tool_args.get("command", ""))
            if _contains_github_token_literal(command):
                reasons.append("token_literal_in_shell_command")
            if reasons:
                reason_text = ",".join(list(dict.fromkeys(reasons)))
                message = f"BLOCKED by github_app guard: {reason_text}"
                safe_message = message.replace('"', "'").replace("\n", " ")
                tool_args["command"] = f'echo "{safe_message}"'
        elif tool_name == "astrbot_execute_python":
            code = str(tool_args.get("code", ""))
            if _contains_github_token_literal(code):
                reasons.append("token_literal_in_python_code")
            if reasons:
                reason_text = ",".join(list(dict.fromkeys(reasons)))
                message = f"BLOCKED by github_app guard: {reason_text}"
                tool_args["code"] = f"print({message!r})"

        if reasons:
            logger.warning(
                f"[GitHubApp] blocked risky tool call: tool={tool_name}, reasons={reasons}"
            )

    @filter.llm_tool(name=GITHUB_CREATE_LICENSE_PR_TOOL_NAME)
    async def github_create_license_pr(
        self,
        event: AstrMessageEvent,
        repo: str = "",
        issue_number: int = 0,
        platform_id: str = "",
        branch_name: str = "",
        license_type: str = "MIT",
        pr_title: str = "",
        pr_body: str = "",
    ) -> str:
        """受控工具：在仓库创建 LICENSE 并发起 PR，不向模型暴露 token。

        Args:
            repo(string): 目标仓库，格式 owner/repo；为空时从当前 GitHub 会话自动解析。
            issue_number(number): 可选，关联 issue 编号；<=0 时自动从会话解析。
            platform_id(string): 可选，当存在多个 github_app 平台时可指定。
            branch_name(string): 可选，目标分支名。
            license_type(string): 许可证类型，当前仅支持 MIT。
            pr_title(string): 可选，PR 标题。
            pr_body(string): 可选，PR 描述。
        """
        if event.get_platform_name() != GITHUB_ADAPTER_TYPE:
            return "该工具仅在 github_app 平台会话中可用。"

        runtime_cfg = get_runtime_plugin_config()
        if not bool(runtime_cfg.get("enable_direct_repo_write_tool", False)):
            return (
                "direct repo write tool is disabled. "
                "Please set enable_direct_repo_write_tool=true."
            )

        adapter = self._resolve_github_adapter(event, platform_id)
        if adapter is None:
            return "未找到可用的 github_app 平台适配器。"
        if not hasattr(adapter, "create_license_pr_for_skill"):
            return "当前 github_app 适配器不支持受控 PR 工具，请升级插件。"

        repo_value = str(repo or "").strip()
        if not repo_value:
            repo_value = str(event.get_extra("github_repository", "")).strip()
        if not repo_value:
            repo_value = _extract_repo_from_github_session(
                str(event.get_extra("github_session_id", "")).strip()
            )
        if not repo_value and event.get_platform_name() == GITHUB_ADAPTER_TYPE:
            repo_value = _extract_repo_from_github_session(event.get_session_id())
        if not repo_value:
            return "缺少 repo 参数，格式应为 owner/repo。"

        try:
            resolved_issue_number = int(issue_number or 0)
        except Exception:
            resolved_issue_number = 0
        if resolved_issue_number <= 0:
            resolved_issue_number = 0
        if resolved_issue_number <= 0:
            resolved_issue_number = _extract_issue_number_from_github_session(
                str(event.get_extra("github_session_id", "")).strip()
            )
        if resolved_issue_number <= 0 and event.get_platform_name() == GITHUB_ADAPTER_TYPE:
            resolved_issue_number = _extract_issue_number_from_github_session(
                event.get_session_id()
            )
        if resolved_issue_number <= 0:
            resolved_issue_number = None

        ok, payload = await adapter.create_license_pr_for_skill(
            repo=repo_value,
            issue_number=resolved_issue_number,
            branch_name=branch_name,
            license_type=license_type,
            pr_title=pr_title,
            pr_body=pr_body,
        )
        if not ok:
            detail = str(payload.get("error", "unknown error"))
            stage = str(payload.get("stage", "")).strip()
            if stage:
                return f"创建 LICENSE PR 失败（{stage}）：{detail}"
            return f"创建 LICENSE PR 失败：{detail}"

        pr_url = str(payload.get("pr_url", "")).strip()
        pr_number = int(payload.get("pr_number", 0) or 0)
        target_repo = str(payload.get("repo", repo_value)).strip()
        head_branch = str(payload.get("head_branch", "")).strip()
        base_branch = str(payload.get("base_branch", "")).strip()
        existing = bool(payload.get("existing_pr", False))

        lines = [
            "LICENSE PR 已创建成功。",
            f"repo: {target_repo}",
            f"base_branch: {base_branch}",
            f"head_branch: {head_branch}",
        ]
        if pr_number > 0:
            lines.append(f"pr_number: {pr_number}")
        if pr_url:
            lines.append(f"pr_url: {pr_url}")
        if existing:
            lines.append("note: 已存在同分支 PR，本次返回已有 PR。")
        return "\n".join(lines)

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

        context_lines = [
            "[GitHub Session Context]",
            f"- repository: {repo_value or '(unknown)'}",
            f"- session_id: {github_session_id or '(unknown)'}",
        ]
        if thread_type:
            context_lines.append(f"- thread_type: {thread_type}")
        if thread_number is not None:
            context_lines.append(f"- thread_number: {thread_number}")
        if thread_title:
            context_lines.append(f"- thread_title: {thread_title}")
        if thread_url:
            context_lines.append(f"- thread_url: {thread_url}")
        if workspace_hint:
            context_lines.append(f"- sandbox_workspace: {workspace_hint}")
            context_lines.append(
                "- note: shell tools will auto-cd to this workspace and auto-clone repo if missing."
            )
        context_lines.append(
            "- instruction: for repository file/code questions, use github_repo_ls first, then github_repo_read / github_repo_search."
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
