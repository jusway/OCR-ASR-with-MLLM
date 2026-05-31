"""PackyCode cc-sale 分组模型配置 —— Claude nothink + adaptive 混合。"""
from core.text_engine import CCSalePackyCodeText

CCSALE_MODELS = { # 不稳定！！！
    # ⚠️ claude-sonnet-4-6 (PackyCode) 流式调用失败 (尝试 4/5): PermissionDeniedError("Error code: 403 - {'error': {'message': 'openai_error', 'type': 'bad_response_status_code', 'param': '', 'code': 'bad_response_status_code'}}")
    "packy-cc-sale-claude-sonnet-4-6-nothink": (CCSalePackyCodeText, {
        "model_name": "claude-sonnet-4-6",   "group": "cc-sale",
        "enable_thinking": False,
    }),
    "packy-cc-sale-claude-sonnet-4-6-adaptive-low": (CCSalePackyCodeText, {
        "model_name": "claude-sonnet-4-6",   "group": "cc-sale",
        "thinking_effort": "low",
    }),
    "packy-cc-sale-claude-sonnet-4-6-adaptive-medium": (CCSalePackyCodeText, {
        "model_name": "claude-sonnet-4-6",   "group": "cc-sale",
        "thinking_effort": "medium",
    }),
    "packy-cc-sale-claude-sonnet-4-6-adaptive-high": (CCSalePackyCodeText, {
        "model_name": "claude-sonnet-4-6",   "group": "cc-sale",
        "thinking_effort": "high",
    }),
    "packy-cc-sale-claude-opus-4-8-nothink": (CCSalePackyCodeText, {
        #⚠️ claude-opus-4-8 (PackyCode) 流式调用失败 (尝试 1/5): PermissionDeniedError("Error code: 403 - {'error': {'message': 'openai_error', 'type': 'bad_response_status_code', 'param': '', 'code': 'bad_response_status_code'}}")
        "model_name": "claude-opus-4-8",     "group": "cc-sale",
        "enable_thinking": False,
    }),
}
