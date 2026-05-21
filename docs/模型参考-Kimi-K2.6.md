# Kimi K2.6 模型参考

> 来源：Kimi 平台官方文档
> 整理日期：2026-05-23

---

## 模型列表

| 模型 | 思考模式 | 推荐用途 |
|------|:--------:|---------|
| `kimi-k2.6` | 默认启用，可禁用 | **推荐使用**，最新最强 |
| `kimi-k2.5` | 默认启用，可禁用 | 同 K2.6 但次强 |
| `kimi-k2-thinking` | 强制启用 | 专门的深度推理 |
| `kimi-k2-turbo-preview` | — | 预览版 |
| `kimi-k2-0905-preview` | — | 预览版 |
| `kimi-k2-thinking-turbo` | 强制启用 | 推理预览版 |

> 没有 Pro/Flash 之分，就是同一个模型开关思考模式。

---

## 核心能力

- **通用 Agent**：Humanity's Last Exam、SWE-Bench Pro、DeepSearchQA 行业领先
- **长程编码**：Rust、Go、Python 等语言，前端/运维/性能优化多场景
- **超长上下文**：256K 窗口
- **多模态**：图片（png/jpeg/webp/gif）、视频（mp4/mpeg/mov/avi 等），body 不超过 100M
- **图片建议**分辨率≤4K (4096×2160)，视频≤2K (2048×1080)，更超分不会提升理解

---

## API 使用要点

### 兼容性

完全兼容 OpenAI SDK，base_url = `https://api.moonshot.cn/v1`。

### 思考模式控制

| 模式 | 方式 |
|------|------|
| 启用思考（默认） | 无需额外参数，或 `extra_body={"thinking": {"type": "enabled"}}` |
| 禁用思考 | `extra_body={"thinking": {"type": "disabled"}}` |

### 跨轮保留思考（Preserved Thinking）

```python
extra_body={"thinking": {"type": "enabled", "keep": "all"}}
```

- `keep: "all"` — 保留历史轮次的 `reasoning_content`，延续思考脉络
- `keep` 不传（默认）— 忽略历史 reasoning，上下文更短更省
- `reasoning_content` 会计入 token 计费

> 与 DeepSeek 的区别：DeepSeek 无工具调用时自动忽略历史 reasoning；Kimi 可以控制保留与否。

### 流式输出处理 `reasoning_content`

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content"):
        reasoning = getattr(delta, "reasoning_content")  # 不要用 .reasoning_content
        # reasoning_content 一定先于 content 出现
    if delta.content:
        # 思考结束，输出最终内容
```

> OpenAI SDK 的 `ChoiceDelta` 类型不提供 `reasoning_content` 字段，只能用 `hasattr`/`getattr` 访问。

### 参数限制

| 参数 | K2.6/K2.5 | kimi-k2-thinking |
|------|-----------|-----------------|
| `thinking` | `enabled`（默认）/ `disabled` | 强制启用 |
| `temperature` | 定值 1.0，**传其他值报错** | 推荐 1.0 |
| `top_p` | 定值 0.95，**传其他值报错** | — |
| `n` | 定值 1，**传其他值报错** | — |
| `presence_penalty` | 定值 0.0，**传其他值报错** | — |
| `frequency_penalty` | 定值 0.0，**传其他值报错** | — |
| `max_tokens` | 默认 32768，思考模式推荐 ≥ 16000 | 推荐 ≥ 16000 |

> **⚠️ 与 DeepSeek 的关键区别**：Kimi 不允许随意设置 temperature/top_p，传了非默认值**直接报错**（DeepSeek 静默忽略）。如果要在 `config/models.py` 中复用，必须确保 Kimi 引擎构造函数不对外暴露这些参数。

### 工具调用约束（思考模式开启时）

- `tool_choice` 只能用 `"auto"` 或 `"none"`
- 多步调用中**必须保留** `reasoning_content` 在上下文
- 联网搜索 `$web_search` 暂不与思考模式兼容

---

## 最佳实践

- 图片 base64 编码，不支持 URL 格式
- 大视频用文件上传
- 思考模式推荐 `max_tokens >= 16000` 以免截断
- 流式输出（`stream=True`）避免网络超时

---

## 常见问题

### Q: 为什么需要保留 `reasoning_content`？

A: 确保多步推理过程中保持连贯性，特别是工具调用场景。服务器自动处理，用户无需手动管理。

### Q: `reasoning_content` 会消耗额外 token 吗？

A: 会计入输入/输出 token 消耗。

### Q: Kimi 有没有 Flash 这种便宜模型？

A: 没有。Kimi K2.6 就一个模型，通过开关思考模式控制行为。
