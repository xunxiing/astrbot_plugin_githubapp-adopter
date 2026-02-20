# astrbot_plugin_githubapp-adapter

用于 AstrBot 的 GitHub App 适配插件。

## 工具

- `github_repo_ls`
  - 只读列目录：按一层目录返回文件与文件夹，支持分页。
- `github_repo_read`
  - 只读读文件：按行分段读取文件内容，适合大文件。
- `github_repo_search`
  - 只读搜索：在仓库内按关键字检索代码路径与片段。

## 安全模型

- GitHub 工具仅在 `github_app` 会话中暴露。
- LLM 可在沙盒中使用本地 shell/git。
- GitHub 会话可自动准备沙盒工作区，并在缺少仓库时自动克隆。
- GitHub 会话会向模型注入仓库、线程标题、线程编号和工作区路径上下文。
- 插件不会向模型返回真实 GitHub 令牌；仅可选下发短期假令牌占位符。
- 工具执行前会将假令牌占位符替换为最小权限的真实只读令牌。
- 可选的令牌字面量防护仅拦截明文 `ghs_...`。

## 关键配置

- `enable_fake_token_bridge=true`（默认）：启用假令牌占位与执行前替换。
- `fake_token_ttl_seconds=900`（默认）：假令牌最大有效期。
- `enable_auto_sandbox_workspace_prepare=true`（默认）：自动准备沙盒工作区。
- `sandbox_workspace_root=/tmp/github-workspaces`：沙盒工作区根目录。
- `sandbox_workspace_clone_depth=1`：自动克隆深度。
- `enforce_tool_write_guard=false`（默认）：是否启用令牌字面量防护总开关。
- `guard_block_token_literal=true`：启用后拦截明文 `ghs_...`。

## Webhook

`http://<astrbot-host>:6185/api/platform/webhook/{webhook_uuid}`
