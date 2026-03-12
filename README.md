# Exa Pool MCP

轻量的 MCP Server，将 [Exa Pool API](https://github.com/chengtx809/exa-pool) / ExaFree 封装为可供 AI 助手调用的工具集。

## 安装与配置

### 1. 下载

下载 `exa_pool_mcp.py` 到本地：

```bash
# 下载到 ~/.claude/ 目录（推荐）
curl -o ~/.claude/exa_pool_mcp.py https://raw.githubusercontent.com/TullyMonster/exa-pool-mcp/main/exa_pool_mcp.py

# 或克隆整个仓库
git clone https://github.com/TullyMonster/exa-pool-mcp.git && cd exa-pool-mcp
```

### 2. 配置

> 以 Claude Code 为例：

```bash
claude mcp add --transport stdio exa-pool --env EXA_POOL_BASE_URL=... --env EXA_POOL_API_KEY=... -- uv run ~/.claude/exa_pool_mcp.py
```

重启 Claude Code 以应用 MCP 的配置变更。

## 工具覆盖

当前版本同时暴露两组工具：

- 官方 Exa MCP 兼容工具：
  - `web_search_exa`
  - `web_search_advanced_exa`
  - `company_research_exa`
  - `people_search_exa`
  - `crawling_exa`
  - `get_code_context_exa`
  - `deep_search_exa`
  - `deep_researcher_start`
  - `deep_researcher_check`
- 兼容旧版调用的工具：
  - `exa_search`
  - `exa_get_contents`
  - `exa_find_similar`
  - `exa_answer`
  - `exa_create_research`
  - `exa_get_research`

说明：

- `get_code_context_exa` 依赖上游暴露 `POST /context`。当前 ExaFree README 公开列出的代理接口不包含 `/context`，因此若后端未扩展该 endpoint，该工具会在后端返回 `404` 或 `405` 时明确提示 code context API 未暴露。
- `exa_answer` 保留了旧版 `include_text` 参数以兼容现有调用方，但当前适配层不会向 `/answer` 透传该字段，以避免部分 Exa Pool 后端返回明显偏离问题的答案。
- 其余官方兼容工具通过现有 `/search`、`/contents`、`/research/v1` 代理能力映射实现。

## 开发与验证

在提交前至少执行：

```bash
python3 -m py_compile exa_pool_mcp.py
python3 -m unittest discover -s tests
```

## ❤️ 致谢与参考

- [Exa Pool GitHub](https://github.com/chengtx809/exa-pool)
- [Exa Docs](https://docs.exa.ai/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)

若需高使用限额或商业用途，请考虑使用 [Exa 官方 API](https://exa.ai/)。
