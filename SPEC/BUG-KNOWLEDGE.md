
## BCE-20260620-001 — heal 从 data-api 路径派生 html id，覆盖合法 canonical id

- **patternId**: BCE-20260620-001
- **layer**: 范式缺陷（工具：gsc-spec heal 序列化）
- **现象**: API `<section>` 经 dom_modify 设为合法 `id="api-*"`（匹配 dataflow 引用的 canonical `API-*`）后，任何 `spec_write`(create/update/delete) 的 auto-chain `fixSpecHtml` 都会把 id 改回 `post-/path` 形态 → 触发 100 个 SPEC error（Invalid ID + dangling DF/CF ref），S2 gate FAIL。
- **根因**: `mcp/src/spec/heal/fixes.mjs` 的 `fixHtmlIdConsistency()` 对 api 元素用 `data-api`（路由路径 "POST /compile"）派生 id，且无幂等守卫 → `dataValue.toLowerCase().replace(/\s+/g,'-')` = `post-/compile`，无条件覆盖人工设定的合法 id。
- **根治模板**（统一策略，已应用于 source repo + plugin cache 6.8.38）:
  1. **idBasis**: api 元素优先用 `data-api-name`（canonical `API-*`）派生 id，而非 `data-api` 路径。SSOT 契约：data-api-name 是规范标识，data-api 仅路由信息。
  2. **char-strip**: 收敛到 HtmlIdSchema，剔除 `/`:`(` 等非法字符 + 首字符保证字母/CJK。
  3. **幂等守卫**: `currentIsValid` —— 当前 id 已 schema-合法则保留，仅缺失/非法才改写。消除 id↔slug 震荡。
- **全量确认**: 41 个 API 元素 / 76 文件横扫，残余 HTTP-path id = 0；S2 gate 7/7 PASS（0 error）；create→delete dataflow 往返 0 error（回归测试）。
- **防复发**: Node 模块缓存使旧进程不重载 —— 修 cache 文件后须 `pkill -f gsc-spec/<ver>/mcp/src/bootstrap.mjs` 杀全部同版本进程，host 重 spawn 才加载新代码。单杀一个会被 host 路由到存活的旧进程。
- **回归断言**: 对任意带 `data-api-name` 的 API 元素，spec_write 往返后 `id` 必须保持 `data-api-name` 的小写形式，不得回退为 `<method>-<path>`。
