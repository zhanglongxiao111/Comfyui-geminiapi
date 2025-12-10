"""
测试 Nano Banana Pro (API Key 版本) 的各项功能
"""
import os
import time
from google import genai
from google.genai import types

# 配置 - 请填入你的 API Key，或设置环境变量 GEMINI_API_KEY
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# 模型配置
IMAGE_MODEL = "gemini-3-pro-image-preview"


def test_image_config_params():
    """测试 ImageConfig 参数是否被支持"""
    print("=" * 60)
    print("测试 1: ImageConfig 参数支持性")
    print("=" * 60)
    
    # 测试 aspect_ratio
    try:
        cfg1 = types.ImageConfig(aspect_ratio="16:9")
        print(f"  ✓ aspect_ratio='16:9' 支持: {cfg1}")
    except Exception as e:
        print(f"  ✗ aspect_ratio 不支持: {e}")
    
    # 测试 image_size
    try:
        cfg2 = types.ImageConfig(image_size="1K")
        print(f"  ✓ image_size='1K' 支持: {cfg2}")
    except Exception as e:
        print(f"  ✗ image_size 不支持: {e}")
    
    # 测试两者组合
    try:
        cfg3 = types.ImageConfig(aspect_ratio="16:9", image_size="1K")
        print(f"  ✓ 组合参数支持: {cfg3}")
    except Exception as e:
        print(f"  ✗ 组合参数不支持: {e}")
    
    # 测试空 ImageConfig
    try:
        cfg4 = types.ImageConfig()
        print(f"  ✓ 空 ImageConfig 支持: {cfg4}")
    except Exception as e:
        print(f"  ✗ 空 ImageConfig 不支持: {e}")
    
    print()


def test_api_connection():
    """测试 API Key 连接是否正常"""
    print("=" * 60)
    print("测试 2: API Key 连接测试")
    print("=" * 60)
    
    if not API_KEY:
        print("  ✗ 未设置 API_KEY，请设置环境变量 GEMINI_API_KEY 或在代码中填入")
        print()
        return False
    
    try:
        client = genai.Client(api_key=API_KEY)
        
        # 使用简单文本请求测试连接
        start_time = time.time()
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Hi, respond with just 'OK'"],
        )
        elapsed = time.time() - start_time
        
        print(f"  ✓ API 连接成功，延迟: {elapsed*1000:.0f}ms")
        print()
        return True
    except Exception as e:
        print(f"  ✗ API 连接失败: {e}")
        print()
        return False


def test_image_generation():
    """测试图像生成功能"""
    print("=" * 60)
    print("测试 3: 图像生成测试 (使用 API Key)")
    print("=" * 60)
    
    if not API_KEY:
        print("  ✗ 未设置 API_KEY，跳过此测试")
        print()
        return
    
    try:
        client = genai.Client(api_key=API_KEY)
        
        # 构建 ImageConfig
        image_config = types.ImageConfig(aspect_ratio="1:1", image_size="1K")
        print(f"  ImageConfig: {image_config}")
        
        config = types.GenerateContentConfig(
            temperature=0.5,
            top_p=0.85,
            response_modalities=["IMAGE"],
            image_config=image_config,
        )
        
        print(f"  正在调用模型: {IMAGE_MODEL}")
        start_time = time.time()
        
        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=["A cute cat sitting on a windowsill"],
            config=config,
        )
        
        elapsed = time.time() - start_time
        print(f"  响应时间: {elapsed:.2f}s")
        
        # 检查响应
        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            print(f"  ✓ 成功生成图像! 大小: {len(inline.data)} bytes")
                            return
        
        print(f"  ✗ 未能从响应中提取图像")
        print(f"  响应内容: {response}")
        
    except Exception as e:
        print(f"  ✗ 图像生成失败: {e}")
    
    print()


def main():
    print()
    print("=" * 60)
    print("Nano Banana Pro (API Key) 测试")
    print("=" * 60)
    print()
    
    # 测试 1: 参数支持性
    test_image_config_params()
    
    # 测试 2: API 连接
    api_ok = test_api_connection()
    
    # 测试 3: 图像生成 (仅在 API 连接成功时)
    if api_ok:
        test_image_generation()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

