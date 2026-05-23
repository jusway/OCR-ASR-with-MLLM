"""
模型配置表 — 两个流水线共用。
加新模型只改这里，不用改入口脚本。
"""
from core.text_engine import DeepSeekText, KimiText

AVAILABLE_MODELS = {
    # ── DeepSeek Pro ──
    "deepseek-pro-nothink": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": False,
    }),
    "deepseek-pro-high": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "high",
    }),
    "deepseek-pro-max": (DeepSeekText, {
        "model_name": "deepseek-v4-pro",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
    # ── DeepSeek Flash ──
    "deepseek-flash-nothink": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": False,
    }),
    "deepseek-flash-high": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": True, "reasoning_effort": "high",
    }),
    "deepseek-flash-max": (DeepSeekText, {
        "model_name": "deepseek-v4-flash",
        "enable_thinking": True, "reasoning_effort": "max",
    }),
    # ── Kimi K2.6（推荐，可开关思考）──
    "kimi-k2.6": (KimiText, {
        "model_name": "kimi-k2.6",
    }),
    "kimi-k2.6-nothink": (KimiText, {
        "model_name": "kimi-k2.6",
        "enable_thinking": False,
    }),
    # ── Kimi K2.5（同 K2.6 但次强）──
    "kimi-k2.5": (KimiText, {
        "model_name": "kimi-k2.5",
    }),
    "kimi-k2.5-nothink": (KimiText, {
        "model_name": "kimi-k2.5",
        "enable_thinking": False,
    }),
}
