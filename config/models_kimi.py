"""Kimi (Moonshot) API 模型配置。"""
from core.text_engine import KimiText

KIMI_MODELS = {
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
