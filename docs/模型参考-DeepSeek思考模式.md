# DeepSeek 思考模式 — API 参考

> 来源：DeepSeek 官方 API 文档
> 整理日期：2026-05-20

---

## 思考模式概述

DeepSeek 模型支持**思考模式（Thinking Mode）**：在输出最终回答之前，模型会先输出一段思维链内容（reasoning_content），以提升最终答案的准确性。

---

## 控制参数

| 含义 | OpenAI 格式 | Anthropic 格式 |
|------|------------|----------------|
| 思考模式开关 | `{"thinking": {"type": "enabled/disabled"}}` | — |
| 思考强度控制 | `{"reasoning_effort": "high/max"}` | `{"output_config": {"effort": "high/max"}}` |

### 说明

1. **默认开关**：思考模式默认为 `enabled`（开启）。
2. **默认 effort**：
   - 普通请求 → `high`
   - 复杂 Agent 类请求（如 Claude Code、OpenCode） → 自动设为 `max`
3. **effort 映射**：
   - `low`、`medium` → 映射为 `high`
   - `xhigh` → 映射为 `max`

### 使用方式（OpenAI SDK）

将 `thinking` 参数传入 `extra_body`：

```python
response = client.chat.completions.create(
  model="deepseek-v4-pro",
  # ...
  reasoning_effort="high",
  extra_body={"thinking": {"type": "enabled"}}
)
```

---

## 输入输出参数

### 不支持的参数（思考模式下忽略）

思考模式下以下参数**不生效**（设置不会报错，但会被 API 忽略）：

- `temperature`
- `top_p`
- `presence_penalty`
- `frequency_penalty`

### 思维链返回值

思维链内容通过 `reasoning_content` 字段返回，与 `content` 同级：

```python
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content
```

---

## 多轮对话拼接

### 无工具调用时

如果一轮对话中**没有工具调用**，则上一轮的 `reasoning_content` **不参与**下一轮的上下文拼接。API 会自动忽略它。

```python
# Turn 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

# Turn 2 — 上一轮的 reasoning_content 被 API 忽略
messages.append(response.choices[0].message)  # 包含 content，不含 reasoning_content
messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=messages,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)
```

### 有工具调用时

如果一轮对话中**进行了工具调用**，则必须**完整回传**该轮的 `reasoning_content` 给 API，否则 API 会返回 400 错误。

```python
# 正确回传方式
messages.append(response.choices[0].message)
# 等价于：
messages.append({
    'role': 'assistant',
    'content': response.choices[0].message.content,
    'reasoning_content': response.choices[0].message.reasoning_content,
    'tool_calls': response.choices[0].message.tool_calls,
})
```

---

## 工具调用示例（完整代码）

```python
import os
import json
from openai import OpenAI
from datetime import datetime

# The definition of the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": "Get the current date",
            "parameters": { "type": "object", "properties": {} },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of a location, the user should supply the location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": { "type": "string", "description": "The city name" },
                    "date": { "type": "string", "description": "The date in format YYYY-mm-dd" },
                },
                "required": ["location", "date"]
            },
        }
    },
]

# The mocked version of the tool calls
def get_date_mock():
    return datetime.now().strftime("%Y-%m-%d")

def get_weather_mock(location, date):
    return "Cloudy 7~13°C"

TOOL_CALL_MAP = {
    "get_date": get_date_mock,
    "get_weather": get_weather_mock
}

def run_turn(turn, messages):
    sub_turn = 1
    while True:
        response = client.chat.completions.create(
            model='deepseek-v4-pro',
            messages=messages,
            tools=tools,
            reasoning_effort="high",
            extra_body={ "thinking": { "type": "enabled" } },
        )
        messages.append(response.choices[0].message)
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        tool_calls = response.choices[0].message.tool_calls
        print(f"Turn {turn}.{sub_turn}\n{reasoning_content=}\n{content=}\n{tool_calls=}")
        # If there is no tool calls, then the model should get a final answer and we need to stop the loop
        if tool_calls is None:
            break
        for tool in tool_calls:
            tool_function = TOOL_CALL_MAP[tool.function.name]
            tool_result = tool_function(**json.loads(tool.function.arguments))
            print(f"tool result for {tool.function.name}: {tool_result}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": tool.id,
                "content": tool_result,
            })
        sub_turn += 1
    print()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url=os.environ.get('DEEPSEEK_BASE_URL'),
)

# Turn 1
turn = 1
messages = [{
    "role": "user",
    "content": "How's the weather in Hangzhou Tomorrow"
}]
run_turn(turn, messages)

# Turn 2
turn = 2
messages.append({
    "role": "user",
    "content": "How's the weather in Guangzhou Tomorrow"
})
run_turn(turn, messages)
```

### 示例输出

```
Turn 1.1
reasoning_content="The user is asking about the weather in Hangzhou tomorrow. I need to get tomorrow's date first, then call the weather function."
content="Let me check tomorrow's weather in Hangzhou for you. First, let me get tomorrow's date."
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_...', function=Function(arguments='{}', name='get_date'), type='function', index=0)]
tool result for get_date: 2026-04-19

Turn 1.2
reasoning_content="Today is 2026-04-19, so tomorrow is 2026-04-20. Now I'll call the weather function for Hangzhou."
content=''
tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_00_...', function=Function(arguments='{"location": "Hangzhou", "date": "2026-04-20"}', name='get_weather'), type='function', index=0)]
tool result for get_weather: Cloudy 7~13°C

Turn 1.3
reasoning_content='The weather result is in. Let me share this with the user.'
content="Here's the weather forecast for **Hangzhou tomorrow (April 20, 2026)**: ..."
tool_calls=None

Turn 2.1
reasoning_content='The user is asking about the weather in Guangzhou tomorrow. Today is 2026-04-19, so tomorrow is 2026-04-20. I can directly call the weather function.'
content=''
tool_calls=[...]
```

---

## 对本项目的实际影响

当前流水线中只有**定位提取原文**步骤使用思考模式（`flash-high`），且没有工具调用，因此：

- `temperature` / `top_p` 传了也被忽略，无需配置
- 单轮对话，无需拼接 `reasoning_content`
- 思维链已通过 `_save_thinking()` 保存为 `思维链_*.md` 文件供人工查阅
