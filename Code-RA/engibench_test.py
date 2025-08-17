import pandas as pd
import json
import requests
import time
from typing import Optional
import re
from datetime import datetime
import logging
import random
from openai import OpenAI
from together import Together
from ollama import Client
import random
import time
import logging
from google import genai
import anthropic

import requests
import os


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_qwen_api(prompt):
    try:
        url = 'https://api.siliconflow.cn/v1/'
        api_key = 'api_key'
        client = OpenAI(base_url=url, api_key=api_key)
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }]
        }
    except Exception as e:
        logger.error(f"GLM API调用错误: {str(e)}")
        return None

# 定义 Gemini API 密钥列表
GEMINI_API_KEYS = [
    'AIzaSyC0xZKNREFUe5_6X27sVj5GoYa9taSppio',
    'AIzaSyDuuq0RQ03rEbHduAgwLNECZs6QgvhHqtc'
]

def call_gemini_api(prompt, retry_attempts=10, retry_delay=30):
    """
    调用 Gemini API 生成内容，支持重试机制。

    参数：
        prompt (str): 用户输入的提示词。
        retry_attempts (int): 最大重试次数。
        retry_delay (int): 重试间隔时间（秒）。

    返回：
        dict 或 None: 包含生成内容的字典，或在失败时返回 None。
    """
    for attempt in range(retry_attempts):
        selected_key = random.choice(GEMINI_API_KEYS)
        try:
            # 配置客户端
            client = genai.Client(api_key=selected_key)
            # 调用 generate_content 方法生成内容
            response = client.models.generate_content(
                # model="gemini-2.0-flash",
                model = "gemini-2.5-flash",
                contents=prompt
            )
            # 提取生成的文本内容
            text = getattr(response, 'text', None)
            if text:
                return {
                    'choices': [{
                        'message': {
                            'content': text
                        }
                    }]
                }
            else:
                logger.error(f"无效的API响应格式: {response}")
        except Exception as e:
            logger.error(f"Gemini API调用错误 (尝试 {attempt + 1}/{retry_attempts}): {str(e)} 使用的Key结尾为 ...{selected_key[-5:]}")
            if attempt < retry_attempts - 1:
                logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error("Gemini API调用达到最大重试次数")
                return None
    return None


def call_deepseek_api(prompt):
    try:
        url = 'https://api.siliconflow.cn/v1/'
        api_key = 'api_key'
        client = OpenAI(base_url=url, api_key=api_key)
        response = client.chat.completions.create(
            # model="Pro/deepseek-ai/DeepSeek-V3",
            model="Pro/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }]
        }
    except Exception as e:
        logger.error(f"GLM API调用错误: {str(e)}")
        return None


def call_openai_api(message):
    api_key = "api_key"
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        store=False,
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return {
        "choices": [{
            "message": {
                "content": completion.choices[0].message.content
            }
        }]
    }

# llama3.3:lastest模型API调用函数
def call_llama_api(prompt):
    try:
        client = Together(api_key="api_key")
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }]
        }
    except Exception as e:
        logger.error(f"Mixtral-8x7B-Instruct-v0.1 API调用错误: {str(e)}")
        return None

def call_glm_api(prompt):
    try:
        client = Together(api_key="api_key")
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}]
        )
        return {
            "choices": [{
                "message": {
                    "content": response.choices[0].message.content
                }
            }]
        }
    except Exception as e:
        logger.error(f"Mixtral-8x7B-v0.1 API调用错误: {str(e)}")
        return None


def call_claude_api(prompt):
    try:
        client = anthropic.Anthropic(
            api_key="api_key"
        )
        
        with client.messages.stream(
            # model="claude-3-7-sonnet-20250219",
            model="claude-3-5-sonnet-20241022",
            # max_tokens=64000,
            max_tokens = 8192,
            messages=[
                {"role": "user", "content": prompt}
            ]
        ) as stream:
            response_text = ""
            for text in stream.text_stream:
                response_text += text
            
            return {
                "choices": [{
                    "message": {
                        "content": response_text
                    }
                }]
            }
    except Exception as e:
        logger.error(f"Claude API调用错误: {str(e)}")
        return None

def call_api(prompt, model_type):
    if model_type.lower() == "gemini":
        return call_gemini_api(prompt)
    elif model_type.lower() == "deepseek":
        return call_deepseek_api(prompt)
    elif model_type.lower() == "gpt-4.1":
        return call_openai_api(prompt)
    elif model_type.lower() == "qwen":
        return call_qwen_api(prompt)
    elif model_type.lower() == "llama":
        return call_llama_api(prompt)
    elif model_type.lower() == "glm":
        return call_glm_api(prompt)
    elif model_type.lower() == "claude":
        return call_claude_api(prompt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def call_answer_model(prompt: str, retry_attempts: int = 10, retry_delay: int = 10, model_type: str = "qwen") -> tuple[Optional[str], Optional[str]]:
    enhanced_prompt = f"""
    Please solve this engineering problem and provide the solution in JSON format:

    Problem:
    {prompt}

    Please provide your solution in the following JSON format:
    {{
        "solution_process": "Your complete solution process",
        "final_answer": "Your answer"
    }}

    Important Notes:
    1. For complex calculations:
       - Keep intermediate steps in fraction form when possible
       - If decimal is necessary, round to 4 decimal places
    2. Show all steps clearly in the solution process
    3. Final answer should be as precise as possible

    Example:
    Problem: "If the airspeed of an airplane is a kilometers per hour and the wind speed is 20 kilometers per hour, what is the difference in kilometers between the distance flown by the airplane against the wind for 3 hours and the distance flown with the wind for 4 hours?"
    
    {{
        "solution_process": "1. With the wind, the effective speed is a + 20 km/h. 2. In 4 hours, the distance flown with the wind is: 4 * (a + 20) = 4a + 80 km. 3. Against the wind, the effective speed is a - 20 km/h. 4. In 3 hours, the distance flown against the wind is: 3 * (a - 20) = 3a - 60 km. 5. The difference in distances is: (4a + 80) - (3a - 60) = 4a + 80 - 3a + 60 = a + 140 km.",
        "final_answer": "a + 140"
    }}

    Ensure your response is in valid JSON format with these exact fields.
    """
    
    for attempt in range(retry_attempts):
        try:
            response = call_api(enhanced_prompt, model_type)
            if response:
                full_response = response['choices'][0]['message']['content']
                try:
                    json_start = full_response.find('{')
                    json_end = full_response.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = full_response[json_start:json_end]
                        json_str = json_str.replace('\\', '\\\\')  # 处理反斜杠
                        json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                        json_str = re.sub(r'[^\x20-\x7E]', '', json_str)
                        
                        try:
                            response_json = json.loads(json_str)
                        except json.JSONDecodeError as je:
                            logger.error(f"JSON解析错误，清理后的字符串: {json_str[:100]}...")
                            if attempt < retry_attempts - 1:
                                time.sleep(retry_delay)
                                continue
                            raise je
                            
                        process = response_json.get('solution_process', '').strip()
                        answer = response_json.get('final_answer', '').strip()
                        
                        if process and answer:
                            return process, answer
                except Exception as e:
                    logger.error(f"处理响应时出错: {str(e)}")
                    if attempt < retry_attempts - 1:
                        time.sleep(retry_delay)
                        continue
            else:
                logger.error("API调用返回空响应")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
                    continue
        except Exception as e:
            logger.error(f"发生错误: {str(e)}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
                continue
    return None, None


def call_comparison_model(prompt: str, retry_attempts: int = 10, retry_delay: int = 30) -> Optional[str]:
    
    for attempt in range(retry_attempts):
        try:
            GEMINI_API_KEYS = [
                'AIzaSyC0xZKNREFUe5_6X27sVj5GoYa9taSppio',
                'AIzaSyDuuq0RQ03rEbHduAgwLNECZs6QgvhHqtc'
            ]

            # 随机选择一个 API Key
            selected_key = random.choice(GEMINI_API_KEYS)  
            # 根据尝试次数选择API密钥
            # current_key = COMPARISON_API_KEYS[attempt % len(COMPARISON_API_KEYS)]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={selected_key}"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            elif response.status_code in [429, 503]:
                logger.warning(f"服务器错误 (状态码: {response.status_code}). 等待 {retry_delay} 秒...")
                time.sleep(retry_delay * 2)
                continue
            else:
                logger.error(f"API调用失败，状态码: {response.status_code}")
                return None


        except Exception as e:
            logger.error(f"发生错误: {str(e)}")
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay)
            else:
                return None
    return None

def compare_answers(generated_answer: str, correct_answer: str) -> Optional[bool]:
    # 最大重试次数
    max_retries = 10
    retry_delay = 30  # 重试间隔时间（秒）
    
    for attempt in range(max_retries):
        if not generated_answer or not correct_answer:
            if attempt < max_retries - 1:
                logger.warning(f"答案为空，第{attempt + 1}次重试...")
                time.sleep(retry_delay)
                continue
            return False
        
        prompt = f"""
        Please analyze these two answers carefully:
        Generated Answer: {generated_answer}
        Standard Answer: {correct_answer}

        Follow these rules for comparison:
        1. For calculation-focused problems:
           - If the numerical values match, consider it correct even if units are missing
           - Focus on the mathematical reasoning and final numerical result
           - Check if the core calculation steps are correct
           - For complex calculations, allow ±2% tolerance in the final numerical result
        
        2. For conceptual or unit-specific problems:
           - Units and their consistency must be considered
           - The complete answer including units is required
        
        3. Consider the answer correct if:
           - The mathematical reasoning is sound
           - The final numerical value matches (within ±2% tolerance for complex calculations)
           - For calculation-focused problems, matching units are not mandatory
        
        Reply only with "True" or "False".
        """
        
        result = call_comparison_model(prompt)
        
        if not result and attempt < max_retries - 1:
            logger.warning(f"API调用失败，第{attempt + 1}次重试...")
            time.sleep(retry_delay)
            continue
        
        result = result.strip().lower() if result else ""
        print(result)
        if 'false' in result:
            return False
        elif 'true' in result:
            return True
        else:
            if attempt < max_retries - 1:
                logger.warning(f"结果不明确，第{attempt + 1}次重试...")
                time.sleep(retry_delay)
                continue
    
    return None


def generated_model(problem: str, model_type: str) -> tuple[Optional[str], Optional[str]]:
    if not problem or problem == "nan":
        return None, None
    return call_answer_model(problem, model_type=model_type)

def save_results(df, generated_model_type, model_type, error_count=None):    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = 'evaluation_process'
    
    # 确保目录存在
    import os
    os.makedirs(base_path, exist_ok=True)
    
    # 保存CSV
    csv_path = f'{base_path}/analyzed2_results_{generated_model_type}_{model_type}_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"结果已保存至CSV: {csv_path}")
    
    # 如果是因为连续错误导致的保存，同时保存XLSX
    if error_count is not None and error_count >= 9:
        xlsx_path = f'{base_path}/analyzed2_results_{generated_model_type}_{model_type}_interrupted_{timestamp}.xlsx'
        df.to_excel(xlsx_path, index=False)
        logger.info(f"检测到连续{error_count}次错误，结果已保存至XLSX: {xlsx_path}")

def generate_statistics(df, model_type, count):
    """生成包含各项正确率的统计信息"""
    return (
        f"当前进度: {count}/{len(df)} 条 ({count/len(df)*100:.1f}%)\n"
        f"原始问题正确率: {df[f'problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"转换问题正确率: {df[f'converted_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        # f"转换问题动态正确率: {df[f'converted_problem_answer_match_dynamic_{model_type}'].mean()*100:.2f}%\n"
        f"知识增强正确率: {df[f'enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"重写问题正确率: {df[f'rewritten_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        f"重写转换问题正确率: {df[f'rewritten_converted_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
        # f"重写转换动态正确率: {df[f'rewritten_converted_problem_answer_match_dynamic_{model_type}'].mean()*100:.2f}%\n"
        f"重写知识增强正确率: {df[f'rewritten_enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%\n"
    )


def main():
    # 让用户选择模型类型
    while True:
        model_choice = input("请选择要使用的模型 (1: Gemini, 2: DeepSeek, 3: Qwen, 4: GPT-4.1, 5: Llama3.3, 6: GLM, 7: Claude): ").strip()
        if model_choice in ['1', '2', '3', '4', '5', '6', '7']:
            model_type = (
                "gemini" if model_choice == '1' else
                ("deepseek" if model_choice == '2' else
                 ("qwen" if model_choice == '3' else
                  ("gpt-4.1" if model_choice == '4' else
                   ("llama" if model_choice == '5' else
                    ("glm" if model_choice == '6' else "claude"))))))
            logger.info(f"已选择使用 {model_type} 模型")
            break
        else:
            print("无效的选择，请输入 1、2、3、4、5、6 或 7")
            
    # 让用户选择生成模型类型
    while True:
        generated_choice = input("请选择要用于生成的模型 (1: Gemini, 2: DeepSeek, 3: Qwen, 4: GPT-4.1, 5: Llama3.3, 6: GLM): ").strip()
        if generated_choice in ['1', '2', '3', '4', '5', '6']:
            generated_model_type = (
                "gemini" if generated_choice == '1' else
                ("deepseek" if generated_choice == '2' else
                 ("qwen" if generated_choice == '3' else
                  ("gpt-4.1" if generated_choice == '4' else
                   ("llama" if generated_choice == '5' else "glm")))))
            logger.info(f"已选择使用 {generated_model_type} 作为生成模型")
            break
        else:
            print("无效的选择，请输入 1、2、3、4、5 或 6")
    
    # 读取CSV文件
    logger.info("开始读取CSV文件...")
    df = pd.read_csv('math_conversion_results2_v3.csv')
    
    # 让用户选择处理范围
    while True:
        try:
            start_line = input("请输入开始处理的行号 (从1开始，直接回车默认为1): ").strip()
            if start_line == "":
                start_line = 1
            else:
                start_line = int(start_line)
            
            if start_line < 1 or start_line > len(df):
                print(f"行号必须在1到{len(df)}之间")
                continue
            break
        except ValueError:
            print("请输入有效的数字")
    
    while True:
        try:
            end_line = input("请输入结束处理的行号 (直接回车默认为最后一行): ").strip()
            if end_line == "":
                end_line = len(df)
            else:
                end_line = int(end_line)
            
            if end_line < start_line or end_line > len(df):
                print(f"结束行号必须在{start_line}到{len(df)}之间")
                continue
            break
        except ValueError:
            print("请输入有效的数字")
    
    # 根据用户选择的范围过滤数据
    df_filtered = df.iloc[start_line-1:end_line].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    
    logger.info(f"将处理第{start_line}行到第{end_line}行的数据，共{len(df_filtered)}条")


    logger.info(f"找到 {len(df_filtered)} 条需要处理的数据")
    
    # 初始化新列
    new_columns = [
        f'problem_llm_process_{model_type}', f'problem_llm_answer_{model_type}', f'problem_answer_match_{model_type}',
        f'converted_problem_llm_process_{model_type}', f'converted_problem_llm_answer_{model_type}', f'converted_problem_answer_match_{model_type}',
        f'enhanced_problem_llm_process_{model_type}', f'enhanced_problem_llm_answer_{model_type}', f'enhanced_problem_answer_match_{model_type}',
        f'rewritten_problem_llm_process_{model_type}', f'rewritten_problem_llm_answer_{model_type}', f'rewritten_problem_answer_match_{model_type}',
        f'rewritten_converted_problem_llm_process_{model_type}', f'rewritten_converted_problem_llm_answer_{model_type}', f'rewritten_converted_problem_answer_match_{model_type}',
        f'rewritten_enhanced_problem_llm_process_{model_type}', f'rewritten_enhanced_problem_llm_answer_{model_type}', f'rewritten_enhanced_problem_answer_match_{model_type}'
    ]
    
    for col in new_columns:
        df_filtered[col] = None
    
    total = len(df_filtered)
    consecutive_errors = 0  # 连续错误计数器
    processed_count = 0  # 新增处理计数器
    
    for idx, row in df_filtered.iterrows():
        processed_count += 1
        actual_line_number = start_line + idx  # 计算在原始文件中的实际行号
        logger.info(f"处理第 {actual_line_number} 行 (进度: {processed_count}/{total})")
        
        try:
            # 处理原始问题
            process, answer = generated_model(row['question'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("处理原始问题失败")
            df_filtered.at[idx, f'problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'problem_answer_match_{model_type}'] = compare_answers(answer, str(row['answer']))

            # 处理转换后的问题
            process, answer = generated_model(row[f'converted_problem_gemini'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("处理转换问题失败")
            df_filtered.at[idx, f'converted_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'converted_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'converted_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['converted_answer_gemini']))
            # df_filtered.at[idx, f'converted_problem_answer_match_dynamic_{model_type}'] = compare_answers(answer, str(row[f'converted_solution_{generated_model_type}']))

            # 处理知识增强的问题
            process, answer = generated_model(row[f'knowledge_enhanced_problem_gemini'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("处理知识增强问题失败")
            df_filtered.at[idx, f'enhanced_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'enhanced_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'enhanced_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['answer']))

            # 处理重写的问题
            process, answer = generated_model(row[f'rewritten_problem_gemini'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("处理重写问题失败")
            df_filtered.at[idx, f'rewritten_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'rewritten_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'rewritten_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['rewritten_answer_gemini']))

            # 处理重写后转换的问题
            process, answer = generated_model(row[f'rewritten_converted_problem_gemini'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("处理重写转换问题失败")
            df_filtered.at[idx, f'rewritten_converted_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'rewritten_converted_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'rewritten_converted_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['rewritten_converted_answer_gemini']))
            # df_filtered.at[idx, f'rewritten_converted_problem_answer_match_dynamic_{model_type}'] = compare_answers(answer, str(row[f'rewritten_converted_solution_{generated_model_type}']))

            # 处理重写后知识增强的问题
            process, answer = generated_model(row[f'rewritten_knowledge_enhanced_problem_gemini'], model_type=model_type)
            if process is None or answer is None:
                raise Exception("处理重写知识增强问题失败")
            df_filtered.at[idx, f'rewritten_enhanced_problem_llm_process_{model_type}'] = process
            df_filtered.at[idx, f'rewritten_enhanced_problem_llm_answer_{model_type}'] = answer
            df_filtered.at[idx, f'rewritten_enhanced_problem_answer_match_{model_type}'] = compare_answers(answer, str(row['rewritten_answer_gemini']))

            consecutive_errors = 0  # 重置连续错误计数
            
        except Exception as e:
            actual_line_number = start_line + idx  # 计算在原始文件中的实际行号
            logger.error(f"处理第 {actual_line_number} 行时发生错误: {str(e)}")
            consecutive_errors += 1
            
            if consecutive_errors >= 9:
                logger.error(f"检测到连续{consecutive_errors}次错误，中断处理并保存结果")
                save_results(df_filtered, generated_model_type, model_type, consecutive_errors)
                return

        # 每处理100条数据保存并输出统计
        if processed_count % 20 == 0:
            stats = generate_statistics(df_filtered, model_type, processed_count)
            logger.info(f"\n=== 进度保存 [{processed_count}/{total}] ===\n{stats}")

        # 每处理100条数据保存并输出统计
        if processed_count % 50 == 0:
            save_results(df_filtered, generated_model_type, model_type)
            stats = generate_statistics(df_filtered, model_type, processed_count)
            logger.info(f"\n=== 进度保存 [{processed_count}/{total}] ===\n{stats}")
        
        time.sleep(5)  # 避免API限制
    
    # 最终保存
    save_results(df_filtered, generated_model_type, model_type)
    
    # 输出统计信息
    logger.info("\n=== 统计信息 ===")
    logger.info(f"总处理数据量: {total}")
    logger.info(f"原始问题正确率: {df_filtered[f'problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"转换问题正确率: {df_filtered[f'converted_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"知识增强问题正确率: {df_filtered[f'enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"重写问题正确率: {df_filtered[f'rewritten_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"重写转换问题正确率: {df_filtered[f'rewritten_converted_problem_answer_match_{model_type}'].mean()*100:.2f}%")
    logger.info(f"重写知识增强问题正确率: {df_filtered[f'rewritten_enhanced_problem_answer_match_{model_type}'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()