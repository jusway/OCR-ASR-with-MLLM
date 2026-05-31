"""模型配置表 — 按引擎拆分为子文件，此文件仅做汇总。"""
from config.models_deepseek import DEEPSEEK_MODELS
from config.models_nvidia import NVIDIA_MODELS
from config.models_kimi import KIMI_MODELS
from config.models_packycode import PACKYCODE_MODELS

AVAILABLE_MODELS = {}
AVAILABLE_MODELS.update(DEEPSEEK_MODELS)
AVAILABLE_MODELS.update(NVIDIA_MODELS)
AVAILABLE_MODELS.update(KIMI_MODELS)
AVAILABLE_MODELS.update(PACKYCODE_MODELS)
