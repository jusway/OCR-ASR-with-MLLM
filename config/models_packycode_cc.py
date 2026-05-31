"""PackyCode cc 分组模型配置 —— Claude nothink 模式。"""
from core.text_engine import CCPackyCodeText

CC_MODELS = {
    # ⚠️ claude-sonnet-4-6 (PackyCode) 流式调用失败 (尝试 1/5): PermissionDeniedError("Error code: 403 - {'error': {'message': '访问被拒绝 (request id: 20260531095856134076548fskjfuoo)', 'type': 'packy_api_error', 'param': '', 'code': 'access_denied'}}")
    "packy-cc-claude-sonnet-4-6-nothink": (CCPackyCodeText, {
        "model_name": "claude-sonnet-4-6",   "group": "cc",
        "enable_thinking": False,
    }),
    "packy-cc-claude-opus-4-8-nothink": (CCPackyCodeText, {
        # ⚠️ claude-opus-4-8 (PackyCode) 流式调用失败 (尝试 3/5): PermissionDeniedError("Error code: 403 - {'error': {'message': '访问被拒绝 (request id: 20260531100138280708460mboqXOV5)', 'type': 'packy_api_error', 'param': '', 'code': 'access_denied'}}")
        "model_name": "claude-opus-4-8",     "group": "cc",
        "enable_thinking": False,
    }),
}
