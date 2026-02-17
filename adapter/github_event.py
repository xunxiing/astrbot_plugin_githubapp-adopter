from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from .session_routing import build_dynamic_session_id, extract_repo_full_name

SUPPORTED_EVENTS = {
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
}


@dataclass(slots=True)
class GitHubParsedEvent:
    event_name: str
    action: str
    repository: str
    session_id: str
    sender_login: str
    sender_id: str
    sender_is_bot: bool
    summary: str
    installation_id: int | None
    timestamp: int


def _get_nested(mapping: Mapping, key: str) -> Mapping:
    value = mapping.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _pick_primary_link(payload: Mapping) -> str:
    for key in ("issue", "pull_request", "comment", "review", "release", "discussion"):
        item = _get_nested(payload, key)
        url = item.get("html_url")
        if isinstance(url, str) and url:
            return url
    if head_commit := _get_nested(payload, "head_commit"):
        url = head_commit.get("url")
        if isinstance(url, str) and url:
            return url
    compare = payload.get("compare")
    if isinstance(compare, str):
        return compare
    return ""


def _pick_subject(event_name: str, payload: Mapping) -> str:
    if event_name in {"issues", "issue_comment"}:
        issue = _get_nested(payload, "issue")
        title = issue.get("title")
        number = issue.get("number")
        if isinstance(title, str) and isinstance(number, int):
            return f"issue #{number} {title}"
    if event_name.startswith("pull_request"):
        pr = _get_nested(payload, "pull_request")
        title = pr.get("title")
        number = pr.get("number")
        if isinstance(title, str) and isinstance(number, int):
            return f"pr #{number} {title}"
    if event_name.startswith("discussion"):
        discussion = _get_nested(payload, "discussion")
        title = discussion.get("title")
        number = discussion.get("number")
        if isinstance(title, str) and isinstance(number, int):
            return f"discussion #{number} {title}"
    if event_name == "push":
        ref = payload.get("ref")
        commits = payload.get("commits")
        if isinstance(ref, str):
            commit_count = len(commits) if isinstance(commits, list) else 0
            return f"{commit_count} commit(s) to {ref}"
    if event_name == "release":
        release = _get_nested(payload, "release")
        name = release.get("name") or release.get("tag_name")
        if isinstance(name, str):
            return f"release {name}"
    return event_name


def parse_github_event(event_name: str, payload: Mapping) -> GitHubParsedEvent | None:
    event_name = (event_name or "").strip().lower()
    if event_name not in SUPPORTED_EVENTS:
        return None

    repository = extract_repo_full_name(payload)
    action = payload.get("action")
    if not isinstance(action, str):
        action = "triggered"

    sender = _get_nested(payload, "sender")
    sender_login = sender.get("login")
    if not isinstance(sender_login, str) or not sender_login:
        sender_login = "unknown"

    sender_id = sender.get("id")
    if isinstance(sender_id, int):
        sender_id = str(sender_id)
    elif not isinstance(sender_id, str) or not sender_id:
        sender_id = sender_login
    sender_type = sender.get("type")
    sender_is_bot = isinstance(sender_type, str) and sender_type.lower() == "bot"
    if not sender_is_bot and isinstance(sender_login, str):
        sender_is_bot = sender_login.lower().endswith("[bot]")

    installation = _get_nested(payload, "installation")
    installation_id = installation.get("id")
    if not isinstance(installation_id, int):
        installation_id = None

    subject = _pick_subject(event_name, payload)
    link = _pick_primary_link(payload)
    summary = (
        f"[GitHub] {repository} {event_name}/{action} by {sender_login}: {subject}"
    )
    if link:
        summary = f"{summary}\n{link}"

    return GitHubParsedEvent(
        event_name=event_name,
        action=action,
        repository=repository,
        session_id=build_dynamic_session_id(event_name, payload),
        sender_login=sender_login,
        sender_id=sender_id,
        sender_is_bot=sender_is_bot,
        summary=summary,
        installation_id=installation_id,
        timestamp=int(time.time()),
    )
