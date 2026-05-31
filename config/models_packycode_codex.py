"""PackyCode codex 分组模型配置 —— GPT 系列。"""
from core.text_engine import CodexPackyCodeText

CODEX_MODELS = {
    "packy-codex-gpt-5.5-nothink": (CodexPackyCodeText, {
        "model_name": "gpt-5.5",
        "group": "codex",
        "enable_thinking": False,
    }),
    "packy-codex-gpt-5.4-nothink": (CodexPackyCodeText, {
        "model_name": "gpt-5.4",
        "group": "codex",
        "enable_thinking": False,
    }),
    "packy-codex-gpt-5.4-high": (CodexPackyCodeText, {
        "model_name": "gpt-5.4-high",
        "group": "codex",
        "enable_thinking": False,
    }),
}
