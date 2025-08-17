def analyze_engineering_problem(problem, answer, model_type="qwen", retry_attempts=1, retry_delay=5):
    prompt = f"""Analyze this engineering problem and provide a detailed mathematical abstraction and knowledge-enhanced version:

Problem: {problem}
Problem Process: Please describe the process to solve the problem step by step.
Given Answer: {answer}

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
    "rewrite_type": int,  // 0-3
    "rewritten_problem": string,
    "rewritten_process": string,
    "rewritten_answer": string,
    "math_only_solvable": boolean,
    "converted_problem": string,
    "converted_process": string,
    "converted_answer": string,
    "knowledge_enhanced_problem": string,
    "original_answer_correct": boolean,
    "rewritten_converted_problem": string,
    "rewritten_converted_process": string,
    "rewritten_converted_answer": string,
    "rewritten_knowledge_enhanced_problem": string,  // 新增字段
    "problem_process": string  // 新增字段
}}
"""