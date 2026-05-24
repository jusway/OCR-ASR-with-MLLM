"""
模型配置表 — 两个流水线共用。
加新模型只改这里，不用改入口脚本。
"""
from core.text_engine import DeepSeekText, KimiText, NvidiaText

AVAILABLE_MODELS = {
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
    # ── 英伟达 GLM-5.1（通过 Nvidia API 调用）──
    "nvidia-glm-5.1-nothink": (NvidiaText, {
        "model_name": "z-ai/glm-5.1",
        "enable_thinking": False,
    }),
    "nvidia-glm-5.1-thinking": (NvidiaText, {
        "model_name": "z-ai/glm-5.1",
        "enable_thinking": True,
        "thinking_kwargs": {"enable_thinking": True, "clear_thinking": False},
    }),
    # ── 英伟达 DeepSeek（通过 Nvidia API 调用）──
    "nvidia-deepseek-pro-nothink": (NvidiaText, {
        "model_name": "deepseek-ai/deepseek-v4-pro",
        "enable_thinking": False,
    }),
    "nvidia-deepseek-pro-thinking": (NvidiaText, {
        "model_name": "deepseek-ai/deepseek-v4-pro",
        "enable_thinking": True,
    }),
    "nvidia-deepseek-flash-nothink": (NvidiaText, {
        "model_name": "deepseek-ai/deepseek-v4-flash",
        "enable_thinking": False,
    }),
    "nvidia-deepseek-flash-thinking": (NvidiaText, {
        "model_name": "deepseek-ai/deepseek-v4-flash",
        "enable_thinking": True,
    }),
    # ── 英伟达 Kimi K2.6（通过 Nvidia API 调用）──
    "nvidia-kimi-k2.6-nothink": (NvidiaText, {
        "model_name": "moonshotai/kimi-k2.6",
        "enable_thinking": False,
    }),
    "nvidia-kimi-k2.6-thinking": (NvidiaText, {
        "model_name": "moonshotai/kimi-k2.6",
        "enable_thinking": True,
    }),
    # ── Kimi K2.6（推荐，可开关思考）──
    "kimi-k2.6-thinking": (KimiText, {
        "model_name": "kimi-k2.6",
    }),
    "kimi-k2.6-nothink": (KimiText, {
        "model_name": "kimi-k2.6",
        "enable_thinking": False,
    }),
    # ── Kimi K2.5（同 K2.6 但次强）──
    "kimi-k2.5-thinking": (KimiText, {
        "model_name": "kimi-k2.5",
    }),
    "kimi-k2.5-nothink": (KimiText, {
        "model_name": "kimi-k2.5",
        "enable_thinking": False,
    }),
}
