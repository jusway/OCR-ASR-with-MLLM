"""DeepSeek 官方 API 模型配置。"""
from core.text_engine import DeepSeekText

DEEPSEEK_MODELS = {
    # ── DeepSeek Pro ──
    "deepseek-pro-nothink": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": False,
    }),
    "deepseek-pro-nothink-stable": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }),
    "deepseek-pro-thinking-high": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "high",
    }),
    "deepseek-pro-thinking-max": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
    # ── DeepSeek Flash ──
    "deepseek-flash-nothink": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": False,
    }),
    "deepseek-flash-thinking-high": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": True, "reasoning_effort": "high",
    }),
    "deepseek-flash-thinking-max": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
}
