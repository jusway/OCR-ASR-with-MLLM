"""测试 config/models.py 中任意模型的连通性和响应速度。"""
import sys
import os
import time

# ==================== 配置：改这里选模型 ====================
MODEL_KEY = "kimi-k2.6-thinking"
# ===========================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.models import AVAILABLE_MODELS


def test_model(key: str):
    if key not in AVAILABLE_MODELS:
        print(f"❌ 未知模型 key: {key}")
        print(f"   可用: {', '.join(AVAILABLE_MODELS.keys())}")
        return

    model_cls, model_kwargs = AVAILABLE_MODELS[key]
    model_name = model_kwargs.get("model_name", "?")

    print(f"模型 key : {key}")
    print(f"实现类  : {model_cls.__name__}")
    print(f"引擎名  : {model_name}")
    print(f"参数    : {model_kwargs}")
    print()

    try:
        engine = model_cls(**model_kwargs)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    print("正在调用模型（说一句话）...")
    t0 = time.time()
    try:
        result = engine.generate("", "说一个字")
        elapsed = time.time() - t0
        if result and result.strip():
            print(f"✅ 调用成功！耗时 {elapsed:.1f}s")
            print(f"   回复 ({len(result)} 字): {result[:200]}")
        else:
            print(f"⚠️ 调用成功但回复为空！耗时 {elapsed:.1f}s")
        if hasattr(engine, '_last_reasoning_content') and engine._last_reasoning_content:
            print(f"   🧠 思考内容: {len(engine._last_reasoning_content)} 字")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"❌ 调用失败（{elapsed:.1f}s）: {e}")


if __name__ == "__main__":
    test_model(MODEL_KEY)
