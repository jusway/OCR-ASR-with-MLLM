"""直连 API 测试 config/models.py 中任意模型 — 不做重试，真实报错。"""
import sys
import os
import time

# ==================== 配置：改这里选模型 ====================
MODEL_KEY = "deepseek-pro-nothink-stable"
# ===========================================================

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.models import AVAILABLE_MODELS


def test_model(key: str):
    if key not in AVAILABLE_MODELS:
        print(f"❌ 未知模型 key: {key}")
        print(f"   可用:")
        for k in AVAILABLE_MODELS:
            print(f"     {k}")
        return

    model_cls, model_kwargs = AVAILABLE_MODELS[key]
    model_name = model_kwargs.get("model_name", "?")

    print(f"模型 key : {key}")
    print(f"实现类  : {model_cls.__name__}")
    print(f"引擎名  : {model_name}")
    print(f"参数    : {model_kwargs}")
    print()

    engine = model_cls(**model_kwargs)

    print("正在调用模型（说一句话）...")
    t0 = time.time()
    response = engine.client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "说一个字"}],
        max_tokens=256,
        temperature=1.0,
        top_p=1.0,
    )
    elapsed = time.time() - t0
    msg = response.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None)
    if content.strip():
        print(f"✅ 调用成功！耗时 {elapsed:.1f}s")
        print(f"   回复 ({len(content)} 字): {content[:200]}")
    else:
        print(f"⚠️ 调用成功但回复为空！耗时 {elapsed:.1f}s")
    if reasoning:
        print(f"   🧠 思考内容: {len(reasoning)} 字")


if __name__ == "__main__":
    test_model(MODEL_KEY)
