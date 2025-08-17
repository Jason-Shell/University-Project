import pandas as pd
import requests
import json
import time
import random
import logging
from tqdm import tqdm

# --- 配置 ---
# 在此处填入您的Gemini API密钥列表
# 程序会从中随机选择一个使用，以减少单个密钥的请求频率
GEMINI_API_KEYS = [
    "AIzaSyDuuq0RQ03rEbHduAgwLNECZs6QgvhHqtc",
    "AIzaSyC0xZKNREFUe5_6X27sVj5GoYa9taSppio"
    # 您可以根据需要添加更多的API密钥
]

# 输入和输出文件名
INPUT_CSV_FILE = 'math_conversion_results2_v3.csv'  # 更新为实际存在的文件
OUTPUT_CSV_FILE = 'math_solution_from_problem_only.csv' # 修改了输出文件名以反映新的逻辑

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_gemini_for_answer(problem: str, retry_attempts: int = 5, retry_delay: int = 20) -> str:
    """
    通过直接调用REST API来与Gemini模型通信，仅根据问题描述生成最终答案。

    参数:
        problem (str): 问题描述。
        retry_attempts (int): 最大重试次数。
        retry_delay (int): 重试间隔时间（秒）。

    返回:
        str: 模型生成的最终答案，如果失败则返回特定错误信息。
    """
    if not problem or pd.isna(problem):
        return "无效的问题输入"

    # 新的Prompt，只包含问题描述
    prompt = f"""
    Please solve the following engineering problem and provide only the final answer.
    Your response should contain just the numerical value or the final expression, without any explanation, units, or labels like "Final Answer:".

    Problem:
    {problem}

    Final Answer:
    """

    for attempt in range(retry_attempts):
        try:
            selected_key = random.choice(GEMINI_API_KEYS)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={selected_key}"
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": prompt}]}]}

            # 发送POST请求
            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                response_json = response.json()
                return response_json['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                logger.warning(f"API返回错误状态码 {response.status_code} (尝试 {attempt + 1}/{retry_attempts}). 响应: {response.text}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"网络请求时出错 (尝试 {attempt + 1}/{retry_attempts}): {str(e)}")
        except (KeyError, IndexError) as e:
            logger.error(f"解析API响应时出错 (尝试 {attempt + 1}/{retry_attempts}): {str(e)}. 可能是响应格式不正确。")
        except Exception as e:
            logger.error(f"发生未知错误 (尝试 {attempt + 1}/{retry_attempts}): {str(e)}")

        if attempt < retry_attempts - 1:
            logger.info(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        else:
            logger.error("API调用达到最大重试次数，返回错误。")
            return "API_ERROR"
            
    return "NO_RESPONSE"


def main():
    """
    主函数，用于读取CSV，处理数据并保存结果。
    """
    try:
        logger.info(f"开始从文件 '{INPUT_CSV_FILE}' 中读取数据...")
        df = pd.read_csv(INPUT_CSV_FILE)
        logger.info(f"成功读取 {len(df)} 行数据。")
    except FileNotFoundError:
        logger.error(f"错误：输入文件 '{INPUT_CSV_FILE}' 未找到。请确保文件与脚本在同一目录下。")
        return

    # 检查必需的列是否存在
    required_column = 'rewritten_problem_gemini'
    if required_column not in df.columns:
        logger.error(f"错误：输入文件必须包含列: '{required_column}'")
        return

    answers = []
    logger.info("开始处理每一行数据并调用Gemini API获取答案...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="解答进度"):
        problem = row[required_column]
        
        # 调用更新后的函数，只传递问题
        answer = call_gemini_for_answer(problem)
        answers.append(answer)
        
        time.sleep(1) # 每处理一行后暂停1秒，避免过于频繁地调用API

    # 将获取的答案列表作为新列添加到DataFrame中
    df['rewritten_answer_gemini'] = answers
    logger.info("所有行处理完毕。")

    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        logger.info(f"结果已成功保存到 '{OUTPUT_CSV_FILE}'。")
    except Exception as e:
        logger.error(f"保存结果到文件时出错: {e}")


if __name__ == "__main__":
    if 'YOUR_GEMINI_API_KEY_1' in GEMINI_API_KEYS or 'YOUR_GEMINI_API_KEY_2' in GEMINI_API_KEYS:
        logger.error("错误：请在脚本的'GEMINI_API_KEYS'列表中填入您自己的Gemini API密钥。")
    else:
        main()