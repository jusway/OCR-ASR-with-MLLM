"""NVIDIA (英伟达) API 模型配置。"""
from core.text_engine import NvidiaText

NVIDIA_MODELS = {
    # ── GLM-5.1 ──
    "nvidia-glm-5.1-nothink": (NvidiaText, {
        "model_name": "z-ai/glm-5.1",
        "enable_thinking": False,
    }),
    "nvidia-glm-5.1-thinking": (NvidiaText, {
        "model_name": "z-ai/glm-5.1",
        "enable_thinking": True,
        "thinking_kwargs": {"enable_thinking": True, "clear_thinking": False},
    }),
    # ── DeepSeek ──
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
    # ── Kimi K2.6 ──
    "nvidia-kimi-k2.6-nothink": (NvidiaText, {
        "model_name": "moonshotai/kimi-k2.6",
        "enable_thinking": False,
    }),
    "nvidia-kimi-k2.6-thinking": (NvidiaText, {
        "model_name": "moonshotai/kimi-k2.6",
        "enable_thinking": True,
    }),
}
