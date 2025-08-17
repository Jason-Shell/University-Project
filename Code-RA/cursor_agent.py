from google import genai
import time
import json
import random

GOOGLE_API_KEY = []

def llm_chat(prompt, return_json=True, model='gemini', max_retries=3):
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if model in ['gemini', 'gemini-think']:
                # 配置API密钥
                genai.configure(api_key=random.choice("AIzaSyDuuq0RQ03rEbHduAgwLNECZs6QgvhHqtc","AIzaSyBxnhSZnXqRSgBKSNPZ2ijkvps8_QixMHA"))
                # 确定模型名称
                model_name = "gemini-2.0-flash-exp" if model == 'gemini' else "gemini-2.0-flash-thinking-exp-1219"
                # 配置生成参数
                generation_config = {"response_mime_type": "application/json"} if return_json else {}
                # 创建模型实例并生成内容
                gemini_model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                response = gemini_model.generate_content(prompt)
                
                if return_json:
                    try:
                        # 尝试解析JSON
                        response_data = json.loads(response.text)
                        return response_data
                    except json.JSONDecodeError as json_err:
                        raise Exception(f"JSON解析失败: {str(json_err)}")
                else:
                    return response.text

            return None
            
        except Exception as e:
            retry_count += 1
            time.sleep(10)
            if retry_count == max_retries:
                raise Exception(f"调用API失败,已重试{max_retries}次: {str(e)} ")
            print(f"调用API失败,正在进行第{retry_count}次重试...")

response = llm_chat('hello, return json', model='gemini', return_json=True, max_retries=1)
print(f"Response: {response}")