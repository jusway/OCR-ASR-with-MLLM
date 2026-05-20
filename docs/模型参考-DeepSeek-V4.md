# DeepSeek-V4 模型参考

> 来源：DeepSeek 官网公告

---

## 模型列表

| 模型 | model_name | 上下文 | 思考模式 | effort |
|------|-----------|--------|:--:|--------|
| DeepSeek-V4-Pro | `deepseek-v4-pro` | 1M | 支持 | high / max |
| DeepSeek-V4-Flash | `deepseek-v4-flash` | 1M | 支持 | high / max |

旧版 `deepseek-chat` / `deepseek-reasoner` 将于 **2026-07-24** 停止使用。

---

## DeepSeek-V4-Pro

- Agent 能力大幅提高，达到当前开源模型最佳水平
- 世界知识仅次于 Gemini-Pro-3.1，大幅领先其他开源模型
- 推理性能超越所有已公开评测的开源模型

## DeepSeek-V4-Flash

- 推理能力接近 Pro
- 简单任务上与 Pro 旗鼓相当
- 世界知识稍逊于 Pro
- 更快捷、经济的 API 服务

---

## 思考模式

- 均支持非思考模式与思考模式
- 思考模式支持 `reasoning_effort: high / max`
- 复杂 Agent 场景建议 `max`
- 思考模式下 `temperature` / `top_p` 不生效
- `max_tokens` 包含思考 token + 输出 token
