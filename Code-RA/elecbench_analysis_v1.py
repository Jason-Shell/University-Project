import pandas as pd
import requests # type: ignore
import json
import time
import logging
import re
from datetime import datetime
from together import Together
from ollama import Client
from openai import OpenAI
import random
import time
import logging
from google import genai
from datasets import load_dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def call_qwen_api(prompt):
    try:
        # client = Client(host='http://115.190.34.89:11434', timeout=100)
        client = Client(host='http://server1.zhaojunhua.org:11434/', timeout=100)
        response = client.chat(model='qwen2.5:72b', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        if response and 'message' in response and 'content' in response['message']:
            return {
                'choices': [{
                    'message': {
                        'content': response['message']['content']
                    }
                }]
            }
        else:
            logger.error("无效的API响应格式")
            return None
    except Exception as e:
        logger.error(f"Qwen API调用错误: {str(e)}")
        return None

def call_gemini_api(prompt):
    GEMINI_API_KEYS = [
        "AIzaSyDuuq0RQ03rEbHduAgwLNECZs6QgvhHqtc",
        
    ]
    # 随机选择一个 API Key
    selected_key = random.choice(GEMINI_API_KEYS)    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={selected_key}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Gemini API调用错误: {str(e)}")
        return None

def call_deepseek_api(prompt):
    try:
        client = Together(api_key="sk-ef1333bbd8504e268d5c7170eddcd4b5")
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
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
        logger.error(f"DeepSeek API调用错误: {str(e)}")
        return None

def call_openai_api(message):
    api_key = "Your API Key"
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4.1",
        store=True,
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

def call_llama_api(prompt):
    try:
        client = Client(host='http://server1.zhaojunhua.org:11434/', timeout=100)
        response = client.chat(model='llama3.3:latest', messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        if response and 'message' in response and 'content' in response['message']:
            return {
                'choices': [{
                    'message': {
                        'content': response['message']['content']
                    }
                }]
            }
        else:
            logger.error("无效的API响应格式")
            return None
    except Exception as e:
        logger.error(f"Llama API调用错误: {str(e)}")
        return None

def call_api(prompt, model_type="qwen"):
    if model_type.lower() == "gemini":
        return call_gemini_api(prompt)
    elif model_type.lower() == "deepseek":
        return call_deepseek_api(prompt)
    elif model_type.lower() == "gpt-4.1":
        return call_openai_api(prompt)
    elif model_type.lower() == "llama":
        return call_llama_api(prompt)
    else:
        return call_qwen_api(prompt)


def analyze_engineering_problem(problem, options, answer, model_type="qwen", retry_attempts=5, retry_delay=30):
    prompt = f"""
You are an expert in engineering education and exam question design.

========================================
Step 1: Subfield Classification (step by step reasoning)
- Carefully read and analyze the problem, identifying any discipline-related technical terms or solution concepts.
- Match the problem to the most appropriate subfield from the list below (avoid 'others' unless absolutely necessary; provide your step-by-step reasoning in your output):
    - Fundamentals of Power Electronics (e.g. rectifiers, inverters, converters, thyristors, power devices)
    - Motor Control Technology (motor control, speed/torque regulation, drivers)
    - Automatic Control Systems for Electric Drives (state space, transfer functions, observers, control law design)
    - Power System Protection (relays, circuit breakers, grid/line protection)
    - Transient Analysis of Power Systems (system surges, transients, short circuit, dynamic stability)
    - Electrical and Electronic Measurement (measurement of voltage, current, power, instrument error)
    - Electrical Machines (transformers, motors, generators, equivalent circuits)
    - Power Supply and Utilization Technology (distribution, utilization, lighting, loads)
    - others
- Output your step-by-step classification reasoning; set the first JSON key as "subfield".

========================================
Step 2: Fill-in-the-Blank Rewrite (modified & question_modified)
Please determine whether the original question can be naturally rewritten as a **fill-in-the-blank problem** (i.e., with one definitive numerical or short formulaic answer):

1. If **can** be rewritten as a fill-in-the-blank:
    - Reformulate the question into a clear fill-in-the-blank format (e.g., "The RMS value of the secondary current is _______.").
    - Set `"modified": true`.
    - Set `"question_modified"` to the fill-in-the-blank format.
    - Set `"answer"` to the expected answer (the original answer content, as in the input).

2. If **cannot** be naturally rewritten as a fill-in-the-blank:
    - Retain the single-choice format, but **integrate all the options directly into the question text** (e.g., append "A) ..., B) ..., C) ... ..." after the main question, or add an 'Options:' section), making it a self-contained choice question.
    - Set `"modified": false`.
    - Set `"question_modified"` to the revised format with options explicitly listed.
    - Set `"answer"` to the correct option's content (not the letter, but the full answer text string).
    - If input does not include options, simply retain the original question as is.

Step by step reasoning is required before making the decision.

JSON output requirements：
- Add two new keys (right after subfield):  
  `"modified": boolean, // True: is fill-in-the-blank, False: still choice`
  `"question_modified": string // fill-in-the-blank content or integrated choice question`
  `"answer": string // see above rules`

========================================
Step 3: Other outputs
Continue with all previous steps of mathematical abstraction, process conversion, knowledge-enhanced version, as in earlier requirements.

Original input:
- Question: {problem}
- Options: {options}
- Given Answer: {answer}


Return EXACTLY this JSON structure:
{{
  "EE_only": boolean, always true!
  "subfield": string, the subfield of the problem
  "modified": boolean, whether the problem is modified
  "question_modified": string, the modified question
  "rewrite_type": int, // 0-3
  "rewritten_problem": string, the rewritten problem
  "rewritten_process": string, the rewritten process
  "can_convert": boolean, whether the problem can be converted to a math_only problem
  "converted_problem": string, the converted problem
  "converted_process": string, the converted process
  "converted_answer": string, the converted answer
  "knowledge_enhanced_problem": string, the knowledge-enhanced problem
  "original_answer_correct": boolean, whether the original answer is correct
  "rewritten_converted_problem": string, the rewritten converted problem
  "rewritten_converted_process": string, the rewritten converted process
  "rewritten_converted_answer": string, the rewritten converted answer
  "rewritten_knowledge_enhanced_problem": string, the rewritten knowledge-enhanced problem
  "problem_process": string, the problem process
}}
Note: All equations are output in LaTeX format to avoid parsing issues.
Analyze this engineering problem and provide a detailed mathematical abstraction and knowledge-enhanced version:

Please analyze and respond in JSON format following these criteria:

1. Rewriting Suitability: Determine the type (0-3):
   - 0: Non-rewritable (use only when necessary)
   - 1: Modify expressions only
   - 2: Modify numerical values only
   - 3: Modify both expressions and numerical values
   // Note: All rewrites must maintain the original problem logic, engineering context, and reasoning/computational requirements

2. Rewritten Problem: Rewrite the problem according to the type of rewriting suitability above. Make the answer as difficult as possible while ensuring that the answer is correct. (Please rewrite the problem in a way that is radically different from your regular logical structure by: (1) avoiding common reasoning patterns in your model, (2) simulating human expert manual rewriting randomness, and (3) using maximum sentence variation.)
   - If 0, return original problem unchanged
   - If 1, modify expressions only
   - If 2, modify numerical values only
   - If 3, modify both expressions and values

3. Rewritten Solution Process: Provide step-by-step explanation including all reasoning, calculations and logic. Clearly state if answer can be obtained directly through formula substitution (shortest solution path without intermediate steps).

4. Rewritten Answer: Provide correct answer for rewritten problem (only types 2/3 may change)

5. Determine if ORIGINAL problem can be solved with ONLY mathematical knowledge (NO engineering background):
   - False if requires any domain-specific knowledge
   - True if solvable through pure mathematical calculations

6. Convert Version:
 ### Instructions:
a. Remove any domain-specific terms or context.  
b. Keep all numerical values and logic used in the solution.  
c. If any scientific constants, atomic weights, or molar ratios are used, directly extract and use the final numerical values (e.g., atomic mass of Na = 23, water density = 1000).  
d. If the original problem includes limiting reagent logic (i.e., which reactant limits the product in a chemical reaction), use `min(x, y)` or a similar expression to represent it.  
e. Preserve the full chain of calculations step by step.  
f. Express logic using symbolic expressions like `Let x = ...`, `min(x, y)`, `z = x × y`, etc.

⚠️ Make sure:
- You use the correct limiting value when a reaction depends on the smaller of two quantities (e.g., `min(x, y)`).
- You **do not include chemistry terms** like "mole," "atom," or "reaction" in the converted version.
- The final numeric answer must exactly match the original solution.
- If it is a calculation question, just extract the numerical calculation part from the problem-solving steps and directly ask the question. No other knowledge is needed.
Simpler, Better!
    
    Examples:

    Original: "In the reaction: Cl₂ + H₂ → 2HCl, 1 mole of Cl₂ reacts with 2 moles of H₂. How many moles of HCl can be formed?"
    "converted_problem": "Let x = 1 and y = 2. They react in the ratio x : y : z = 1 : 1 : 2. Total product z = min(x, y) * 2. Find the result."
    "converted_process":"Step 1: Let x = 1 and y = 2. Step 2: The reaction ratio is x : y : z = 1 : 1 : 2. Step 3: The limiting reactant is min(x, y) = 1. Step 4: Compute total product z = 1 * 2 = 2. The answer is 2."
    "converted_answer": 2

    Original: "A 2m wide platform sinks 0.01m under 60kg. Estimate its length assuming water density = 1000 kg/m³."
    "converted_problem": "Let x = 60 / (2 * 0.01 * 1000). Calculate the value of x."
    "converted_process": "Let x = 60 / (2 * 0.01 * 1000). Step 1: Compute denominator = 2 * 0.01 * 1000 = 20. Step 2: Compute x = 60 / 20 = 3. The answer is 3."
    "converted_answer": 3.0


7. Knowledge-Enhanced Version:
    ⚠️ Make sure the final numerical answer to the converted mathematical problem is exactly the same as the original problem.
    Given:
    - List all relevant formulas or principles (e.g., Ohm's Law: V = I × R)
    - Include physical constants with values if they are involved (e.g., g = 9.8 m/s²)
    - Specify unit conversions if applicable (e.g., 1 kWh = 3.6 × 10⁶ J)
    - State any assumptions or ideal conditions if necessary (e.g., assume no heat loss)

    Problem:
    Repeat the original question exactly as stated

   Example:
   Original: "Calculate voltage across 5Ω resistor with 2A current"
   Enhanced:
   "Given:
   - Ohm's Law: V = I * R
   - Problem: Calculate voltage across 5Ω resistor with 2A current"

Response format:
{{
  "EE_only": boolean, always true!
  "subfield": string, the subfield of the problem
  "modified": boolean, whether the problem is modified
  "question_modified": string, the modified question
  "rewrite_type": int, // 0-3
  "rewritten_problem": string, the rewritten problem
  "rewritten_process": string, the rewritten process
  "can_convert": boolean, whether the problem can be converted to a gemini problem
  "converted_problem": string, the converted problem
  "converted_process": string, the converted process
  "converted_answer": string, the converted answer
  "knowledge_enhanced_problem": string, the knowledge-enhanced problem
  "original_answer_correct": boolean, whether the original answer is correct
  "rewritten_converted_problem": string, the rewritten converted problem
  "rewritten_converted_process": string, the rewritten converted process
  "rewritten_converted_answer": string, the rewritten converted answer
  "rewritten_knowledge_enhanced_problem": string, the rewritten knowledge-enhanced problem
  "problem_process": string, the problem process
}} Note: All equations are output in LaTeX format to avoid parsing issues

"""
    
    for attempt in range(retry_attempts):
        try:
            response = call_api(prompt, model_type)
            if response:
                try:
                    # 根据不同模型处理返回结果
                    if model_type.lower() == "gemini":
                        content = response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    elif model_type.lower() == "deepseek":
                        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    elif model_type.lower() == "openai":
                        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:
                        content = (response.get('choices', [{}])[0]
                                .get('message', {})
                                .get('content') or
                                response.get('choices', [{}])[0]
                                .get('message', {})
                                .get('reasoning_content', ''))
                    
                    if not content:
                        logger.error("API 返回内容为空")
                        if attempt < retry_attempts - 1:
                            logger.info(f"尝试重试 {attempt + 1}/{retry_attempts}")
                            time.sleep(retry_delay)
                            continue
                        return None
                    
                    # 尝试解析 JSON 内容
                    try:
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        if json_start != -1 and json_end != -1:
                            json_str = content[json_start:json_end]
                            # 保留数学符号，只替换换行和制表符
                            json_str = json_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                            # 修改正则表达式以保留数学符号
                            json_str = re.sub(r'[^\x20-\x7E\u0080-\uffff]', '', json_str)
                            return json.loads(json_str)
                        else:
                            logger.error("无法在响应中找到 JSON 内容")
                            if attempt < retry_attempts - 1:
                                logger.info(f"尝试重试 {attempt + 1}/{retry_attempts}")
                                time.sleep(retry_delay)
                                continue
                            return None
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 解析错误: {str(e)}")
                        if attempt < retry_attempts - 1:
                            logger.info(f"尝试重试 {attempt + 1}/{retry_attempts}")
                            time.sleep(retry_delay)
                            continue
                        return None
                        
                except Exception as e:
                    logger.error(f"处理 API 响应时出错: {str(e)}")
                    if attempt < retry_attempts - 1:
                        logger.info(f"尝试重试 {attempt + 1}/{retry_attempts}")
                        time.sleep(retry_delay)
                        continue
                    return None
            else:
                logger.error("API 调用返回空响应")
                if attempt < retry_attempts - 1:
                    logger.info(f"尝试重试 {attempt + 1}/{retry_attempts}")
                    time.sleep(retry_delay)
                    continue
                return None
                
        except Exception as e:
            logger.error(f"API 调用出错: {str(e)}")
            if attempt < retry_attempts - 1:
                logger.info(f"尝试重试 {attempt + 1}/{retry_attempts}")
                time.sleep(retry_delay)
                continue
            return None
            
    return None

def main():
    # 选择使用的模型
    while True:
        model_choice = input("请选择要使用的模型 (1: Gemini, 2: DeepSeek, 3: Qwen, 4: GPT-4.1, 5: Llama): ").strip()
        if model_choice in ['1', '2', '3', '4', '5']:
            model_type = (
                "gemini" if model_choice == '1' else
                ("deepseek" if model_choice == '2' else
                 ("qwen" if model_choice == '3' else
                  ("gpt-4.1" if model_choice == '4' else "llama")))
            )
            logger.info(f"已选择使用 {model_type} 模型")
            break
        else:
            print("无效的选择，请输入 1、2、3、4 或 5")
    
    # 加载数据集
    df = load_dataset("m-a-p/SuperGPQA")
    
    # 过滤出满足条件的数据
    engineering_data = df.filter(
        lambda x: x['field'] == 'Electrical Engineering' and 
                  x['is_calculation'] == True 
    )
    
    # 获取具体的数据集
    engineering_dataset = engineering_data['train']  # 假设你要处理的是训练集
    
    # # 选择前三行进行测试
    eng_df = engineering_dataset
    #.select(range(10))
    
    # 将 Dataset 转换为 Pandas DataFrame
    eng_df = eng_df.to_pandas()
    eng_df =  eng_df.iloc[:]
    
    # 输出测试数据
    print(eng_df)

    # 添加新列（根据选择的模型调整列名）
    new_columns = [
        f'EE_only_{model_type}',
        f'subfield_{model_type}',
        f'modified_{model_type}',
        f'question_modified_{model_type}',
        f'rewrite_type_{model_type}',
        f'rewritten_problem_{model_type}',
        f'rewritten_process_{model_type}',
        f'can_convert_{model_type}',
        f'converted_problem_{model_type}',
        f'converted_process_{model_type}',
        f'converted_answer_{model_type}',
        f'knowledge_enhanced_problem_{model_type}',
        f'original_answer_correct_{model_type}',
        f'rewritten_converted_problem_{model_type}', 
        f'rewritten_converted_process_{model_type}', 
        f'rewritten_converted_answer_{model_type}',  
        f'rewritten_knowledge_enhanced_problem_{model_type}', 
        f'problem_process_{model_type}' 
    ]
    
    # 初始化新列为 None
    for col in new_columns:
        eng_df[col] = None
    
    error_count = 0
    max_errors = 5  # 最大错误次数改为5次
    processed_count = 0  # 记录已处理的数据条数
    
    for idx, row in eng_df.iterrows():
        try:
            analysis = analyze_engineering_problem(
                row['question'],
                row['options'],
                row['answer'],
                model_type=model_type
            )
            
            if analysis:
                # 将分析结果添加到原始数据中
                for key in ['EE_only', 'subfield', 'modified', 'question_modified', 
                          'rewrite_type', 'rewritten_problem', 'rewritten_process',
                          'can_convert', 'converted_problem', 'converted_process', 'converted_answer',
                          'knowledge_enhanced_problem', 'original_answer_correct',
                          'rewritten_converted_problem', 'rewritten_converted_process', 'rewritten_converted_answer',
                          'rewritten_converted_answer', 'rewritten_knowledge_enhanced_problem',
                          'problem_process']: 
                    eng_df.loc[idx, f"{key}_{model_type}"] = analysis.get(key)
                
                processed_count += 1
                error_count = 0  # 重置错误计数
                logger.info(f"成功处理问题 {idx+1}")
            else:
                error_count += 1
                logger.warning(f"处理问题 {idx+1} 返回None，跳过处理下一条数据")
                if error_count >= max_errors:
                    logger.warning(f"连续{max_errors}次错误，保存当前结果并继续处理下一条数据")
                    error_count = 0  # 重置错误计数
                continue
            
            # 每处理10条数据保存一次
            if processed_count % 100 == 0:
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = f'elecbench_analysis_results_superGPQA_{model_type}_{processed_count}_{current_time}.csv'
                eng_df.to_csv(save_path, index=False)
                logger.info(f"已处理{processed_count}条数据，保存阶段性结果至: {save_path}")
            
        except Exception as e:
            logger.error(f"处理数据时出错: {str(e)}")
            error_count += 1
            
            # 如果错误次数达到上限，保存当前结果并退出
            if error_count >= max_errors:
                logger.warning(f"错误次数达到{max_errors}次，保存当前结果并退出")
                current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f'elecbench_analysis_results_superGPQA_{model_type}.csv'
                eng_df.to_csv(output_path, index=False)
                logger.info(f"部分结果已保存至: {output_path}")
                return
    
    # 保存完整结果
    output_path = f'elecbench_analysis_results_superGPQA_{model_type}.csv'
    eng_df.to_csv(output_path, index=False)
    logger.info(f"处理完成，结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
