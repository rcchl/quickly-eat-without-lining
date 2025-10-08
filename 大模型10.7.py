import requests
import os
import json
from datetime import datetime
import re


def load_json_file(file_path):
    """
    从JSON文件加载内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载JSON文件: {file_path}")
        return data
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None


def extract_key_information_from_json(json_data, api_key):
    """
    使用DeepSeek从JSON数据中提取有效信息并生成关联词
    """
    api_url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 将JSON数据转换为字符串用于提示词
    json_text = json.dumps(json_data, ensure_ascii=False, indent=2)

    # 构建提示词
    prompt = f"""
    请分析以下JSON数据，识别其中的有效信息和关键字段，并为每个有效信息生成10-20个相关关联词。

    JSON数据：
    {json_text}

    要求：
    1. 首先识别JSON中的有效信息（如公司名称、产品信息、数字数据、日期、人员等关键字段）
    2. 为每个有效信息生成10-20个相关的关联词、同义词或扩展术语
    3. 返回格式必须是严格的JSON格式：
    {{
        "extracted_key_info": [
            {{
                "key_info": "识别出的有效信息1",
                "related_words": ["关联词1", "关联词2", "关联词3", ...]
            }},
            {{
                "key_info": "识别出的有效信息2", 
                "related_words": ["关联词1", "关联词2", "关联词3", ...]
            }}
        ]
    }}
    4. 确保每个有效信息都有对应的关联词列表
    5. 只返回JSON格式数据，不要有其他解释
    """

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.3
    }

    try:
        print("正在分析JSON数据并生成关联词...")
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        response_text = result['choices'][0]['message']['content']

        # 尝试从响应中提取JSON数据
        try:
            # 查找JSON格式的内容
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                related_data = json.loads(json_str)
                return related_data
            else:
                # 如果找不到JSON，尝试直接解析整个响应
                related_data = json.loads(response_text)
                return related_data
        except json.JSONDecodeError:
            print("无法解析模型返回的JSON格式")
            print(f"原始响应: {response_text}")
            return None

    except Exception as e:
        print(f"生成关联词失败: {e}")
        if hasattr(e, 'response'):
            print(f"响应状态码: {e.response.status_code}")
            print(f"响应内容: {e.response.text}")
        return None


def save_final_results(original_data, key_info_results, output_path):
    """
    将最终结果保存为JSON格式
    """
    result_data = {
        "metadata": {
            "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": "C:\\Users\\thy\\Desktop\\ocr_result.json",
            "analysis_type": "key_information_extraction"
        },
        "original_data_preview": {
            "data_type": str(type(original_data)),
            "keys": list(original_data.keys()) if isinstance(original_data, dict) else "Not a dictionary"
        },
        "extraction_results": key_info_results,
        "statistics": {
            "total_key_info_items": len(key_info_results.get('extracted_key_info', [])),
            "total_related_words": sum(
                len(item.get('related_words', []))
                for item in key_info_results.get('extracted_key_info', [])
            )
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return result_data


def main():
    # 使用您提供的API密钥
    api_key = "sk-2077121e9c75425e986806f435854d2d"

    # JSON文件路径
    json_file_path = r"C:\Users\thy\Desktop\ocr_result.json"

    print(f"当前工作目录: {os.getcwd()}")
    print(f"要处理的JSON文件: {json_file_path}")

    # 检查JSON文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：JSON文件不存在 - {json_file_path}")
        return

    # 步骤1: 加载JSON文件
    json_data = load_json_file(json_file_path)

    if not json_data:
        print("无法加载JSON数据，程序退出")
        return

    print("JSON数据加载成功！")
    print("=" * 50)

    # 显示JSON数据的预览
    if isinstance(json_data, dict):
        print("JSON数据结构预览：")
        print(f"顶层键: {list(json_data.keys())}")
        # 显示前几个键值对作为预览
        preview_items = list(json_data.items())[:3]
        for key, value in preview_items:
            print(f"  {key}: {str(value)[:100]}...")
    else:
        print(f"数据格式: {type(json_data)}")
        print(f"数据预览: {str(json_data)[:200]}...")

    print("=" * 50)

    # 步骤2: 使用DeepSeek分析JSON数据并生成关联词
    key_info_results = extract_key_information_from_json(json_data, api_key)

    if key_info_results:
        print("关键信息提取和关联词生成成功！")
        print("=" * 50)

        # 显示提取结果
        extracted_items = key_info_results.get('extracted_key_info', [])
        print(f"共提取到 {len(extracted_items)} 个有效信息：")

        for i, item in enumerate(extracted_items, 1):
            key_info = item.get('key_info', '未知信息')
            related_words = item.get('related_words', [])
            print(f"\n{i}. 有效信息: {key_info}")
            print(
                f"   关联词 ({len(related_words)}个): {', '.join(related_words[:5])}{'...' if len(related_words) > 5 else ''}")

        # 保存结果到JSON文件
        result_file = "key_information_analysis.json"
        final_result = save_final_results(json_data, key_info_results, result_file)

        print("=" * 50)
        print(f"结果已保存到JSON文件: {os.path.abspath(result_file)}")

        # 显示统计信息
        stats = final_result["statistics"]
        print(f"\n统计信息：")
        print(f"有效信息数量: {stats['total_key_info_items']}个")
        print(f"总关联词数量: {stats['total_related_words']}个")

    else:
        print("关键信息提取失败")


if __name__ == "__main__":
    main()
    print("程序执行完毕")