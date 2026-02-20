from __future__ import annotations

import re
from typing import Any


def normalize_repo_full_name(value: Any) -> str:
    raw = str(value or "").strip().strip("/")
    if "/" not in raw:
        return ""
    owner, repo = raw.split("/", 1)
    owner = owner.strip()
    repo = repo.strip()
    if not owner or not repo:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        return ""
    return f"{owner}/{repo}"


def sanitize_workspace_session_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "default"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-")
    if not cleaned:
        return "default"
    return cleaned[:80]


def _escape_shell_dq(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("$", "\\$")
    text = text.replace("`", "\\`")
    return text


def build_shell_workspace_bootstrap_command(
    *,
    command: str,
    repo: str,
    session_key: str,
    workspace_root: str,
    clone_depth: int,
) -> str:
    repo_escaped = _escape_shell_dq(repo)
    session_escaped = _escape_shell_dq(session_key)
    root_escaped = _escape_shell_dq(workspace_root)
    depth = clone_depth if clone_depth > 0 else 1
    original = str(command or "").rstrip()
    if not original:
        original = "pwd"

    return (
        f'REPO="{repo_escaped}"\n'
        f'WORK_ROOT="{root_escaped}"\n'
        f'WORK_SESSION="{session_escaped}"\n'
        'OWNER="${REPO%%/*}"\n'
        'NAME="${REPO##*/}"\n'
        'WS="${WORK_ROOT}/${OWNER}__${NAME}/${WORK_SESSION}"\n'
        'CLONE_URL="https://github.com/${REPO}.git"\n'
        'reclone_needed=0\n'
        'if [ -d "${WS}/.git" ]; then\n'
        '  if ! git -C "${WS}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then\n'
        "    reclone_needed=1\n"
        "  fi\n"
        "fi\n"
        'if [ "${reclone_needed}" = "1" ]; then\n'
        '  rm -rf "${WS}"\n'
        "fi\n"
        'if [ ! -d "${WS}/.git" ]; then\n'
        '  mkdir -p "$(dirname "${WS}")"\n'
        f'  if ! git clone --depth {depth} --filter=blob:none "${{CLONE_URL}}" "${{WS}}" >/dev/null 2>&1; then\n'
        '    echo "[github-workspace] clone failed for ${REPO}" >&2\n'
        '    mkdir -p "${WS}"\n'
        "  fi\n"
        "fi\n"
        'if [ -d "${WS}/.git" ] && ! git -C "${WS}" rev-parse --verify HEAD >/dev/null 2>&1; then\n'
        f'  git -C "${{WS}}" fetch --depth {depth} origin >/dev/null 2>&1 || true\n'
        '  BR="$(git -C "${WS}" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed \'s#^origin/##\')"\n'
        '  if [ -z "${BR}" ]; then\n'
        '    BR="$(git -C "${WS}" for-each-ref --format=\'%(refname:short)\' refs/remotes/origin 2>/dev/null | sed -n \'s#^origin/##p\' | head -n 1)"\n'
        "  fi\n"
        '  if [ -n "${BR}" ]; then\n'
        '    git -C "${WS}" checkout -B "${BR}" "origin/${BR}" >/dev/null 2>&1 || true\n'
        "  fi\n"
        "fi\n"
        'if [ -d "${WS}/.git" ] && ! git -C "${WS}" rev-parse --verify HEAD >/dev/null 2>&1; then\n'
        '  rm -rf "${WS}"\n'
        '  mkdir -p "$(dirname "${WS}")"\n'
        f'  if ! git clone --depth {depth} --filter=blob:none "${{CLONE_URL}}" "${{WS}}" >/dev/null 2>&1; then\n'
        '    echo "[github-workspace] clone retry failed for ${REPO}" >&2\n'
        '    mkdir -p "${WS}"\n'
        "  fi\n"
        "fi\n"
        'if [ -d "${WS}/.git" ] && ! git -C "${WS}" rev-parse --verify HEAD >/dev/null 2>&1; then\n'
        '  echo "[github-workspace] no checked out commit (repo may be empty or clone incomplete)" >&2\n'
        "fi\n"
        'cd "${WS}" 2>/dev/null || true\n'
        f"{original}"
    )
