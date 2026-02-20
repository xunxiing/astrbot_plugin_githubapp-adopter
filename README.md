# astrbot_plugin_githubapp-adopter

GitHub App adapter plugin for AstrBot.

## Tools

- `github_create_license_pr`
  - Controlled remote write flow: create/reuse branch, write `LICENSE` (MIT), open PR.
  - Token is handled inside plugin only and never exposed to LLM.
- `github_repo_ls`
  - List one directory level with pagination (read-only).
- `github_repo_read`
  - Read file content by line range (chunked, read-only).
- `github_repo_search`
  - Search keyword hits in repository code (read-only).
  - Uses GitHub App installation token inside plugin; token is never exposed to LLM.

## Removed

- Legacy token-issuing tool has been removed and is no longer exposed to LLM.

## Security Model

- GitHub tools are exposed only for `github_app` session requests.
- LLM can use local shell/git in sandbox.
- For GitHub sessions, plugin can auto-prepare session workspace and auto-clone target repo in sandbox.
- Remote write (branch/push/PR) is handled by controlled plugin tools only.
- Repo / thread title / session workspace path are injected into LLM system prompt for GitHub sessions.
- Plugin does not return real GitHub tokens to model.
- Optional literal token guard only blocks plain `ghs_...` tokens in shell/python tool args.

## Key Config

- `enable_direct_repo_write_tool=true`: allow `github_create_license_pr`.
- `enable_auto_sandbox_workspace_prepare=true` (default): auto-bootstrap sandbox workspace for shell calls.
- `sandbox_workspace_root=/tmp/github-workspaces`: workspace root in sandbox.
- `sandbox_workspace_clone_depth=1`: shallow clone depth for auto bootstrap.
- `enforce_tool_write_guard=false` (default): do not restrict local shell/git behavior.
- `guard_block_token_literal=true`: if guard is enabled, block plain token literals.

## Webhook

`http://<astrbot-host>:6185/api/platform/webhook/{webhook_uuid}`
