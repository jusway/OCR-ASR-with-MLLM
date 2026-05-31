"""PackyCode 中转站模型配置（汇总）。按分组拆分为子文件，此文件仅做汇总。"""
from config.models_packycode_codex import CODEX_MODELS
from config.models_packycode_cc import CC_MODELS
from config.models_packycode_awsq import AWSQ_MODELS
from config.models_packycode_ccsale import CCSALE_MODELS

PACKYCODE_MODELS = {}
PACKYCODE_MODELS.update(CODEX_MODELS)
PACKYCODE_MODELS.update(CC_MODELS)
PACKYCODE_MODELS.update(AWSQ_MODELS)
PACKYCODE_MODELS.update(CCSALE_MODELS)
