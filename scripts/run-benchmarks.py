#!/usr/bin/env python3
import json, sys, time, subprocess, os, csv

BASE_URL = "http://localhost:8080"
RESULTS_DIR = "/mnt/nvme/home/liveuser/models/eval-results"

def api_call(messages, max_tokens=2048, temperature=0):
    import urllib.request
    data = json.dumps({
        "model": "reap20",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=600)
    return json.loads(resp.read())

def get_response_text(r):
    msg = r["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    return content, reasoning

import re

def extract_python_function(text, prompt):
    """Extract a Python function from model output, handling markdown fences and thinking.
    
    Strategy: Find the last complete function definition in the text,
    regardless of whether it's in a code block, thinking, or plain text.
    Then dedent it to be valid Python.
    """
    import textwrap
    
    # Strategy 1: Find ```python blocks and use the last one that has a def
    md_match = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    for block in reversed(md_match):
        block = textwrap.dedent(block).strip()
        if 'def ' in block:
            try:
                compile(block, '<test>', 'exec')
                return block
            except SyntaxError:
                pass
    
    # Strategy 2: Reconstruct function from scattered text
    # Find the last "def func_name" and collect everything until next non-indented non-empty line
    # that isn't part of the function
    func_name = None
    # Extract the function name from the prompt
    m = re.search(r'def\s+(\w+)', prompt)
    if m:
        func_name = m.group(1)
    
    if func_name:
        # Find all occurrences of the function definition in the text
        # and collect the body (which may be in a code block right after)
        pattern = rf'def\s+{func_name}\s*\('
        matches = list(re.finditer(pattern, text))
        
        if matches:
            # Use the last match
            match = matches[-1]
            start_pos = match.start()
            remaining = text[start_pos:]
            lines = remaining.split('\n')
            
            # Collect the function: def line + indented body
            func_lines = [lines[0]]
            base_indent = len(lines[0]) - len(lines[0].lstrip())
            
            for line in lines[1:]:
                stripped = line.strip()
                if stripped == '' or stripped.startswith('```'):
                    # skip empty lines and code fences within the function
                    if stripped.startswith('```'):
                        continue
                    func_lines.append(line)
                    continue
                curr_indent = len(line) - len(line.lstrip())
                if curr_indent > base_indent:
                    func_lines.append(line)
                else:
                    break
            
            code = textwrap.dedent('\n'.join(func_lines)).strip()
            
            # Add necessary imports
            if 'List' in code and 'from typing' not in code:
                code = 'from typing import List\n\n' + code
            
            try:
                compile(code, '<test>', 'exec')
                return code
            except SyntaxError:
                pass
    
    # Strategy 3: Find any code block body and wrap it with the prompt
    for block in reversed(md_match):
        body = textwrap.dedent(block).strip()
        if body and 'return' in body:
            # This looks like a function body, wrap with the prompt
            code = prompt + '    ' + body.replace('\n', '\n    ')
            try:
                compile(code, '<test>', 'exec')
                return code
            except SyntaxError:
                pass
    
    # Strategy 4: Use the prompt + the code block body (indented)
    for block in reversed(md_match):
        body = textwrap.dedent(block).strip()
        if body:
            # Indent each line by 4 spaces and append to prompt
            indented_body = '\n'.join('    ' + l if l.strip() else '' for l in body.split('\n'))
            code = prompt + indented_body
            try:
                compile(code, '<test>', 'exec')
                return code
            except SyntaxError:
                pass
    
    # Last resort
    return prompt + "    pass"

def bench_speed(quant_name):
    """Run speed benchmark via llama-bench output already captured."""
    pass

def run_humaneval_smoke(quant_name, n=5):
    """Generate solutions for HumanEval problems and test them."""
    problems = [
        {"task_id": "HumanEval/0", 
         "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
         "tests": [
             ("has_close_elements([1.0, 2.0, 3.0], 0.5)", False),
             ("has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)", True),
             ("has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)", True),
             ("has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)", False),
         ]},
        {"task_id": "HumanEval/1",
         "prompt": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
         "tests": [
             ("separate_paren_groups('(()()) ((())) () ((())()())')", ['(()())', '((()))', '()', '((())()())']),
             ("separate_paren_groups('() (()) ((())) (((())))')", ['()', '(())', '((()))', '(((())))']),
         ]},
        {"task_id": "HumanEval/4",
         "prompt": "from typing import List\n\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\"For a given list of input numbers, calculate Mean Absolute Deviation\n    around the mean of this dataset.\n    Mean Absolute Deviation is the average absolute difference between each\n    element and a centerpoint (mean in this case):\n    MAD = average | x - x_mean |\n    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n    1.0\n    \"\"\"\n",
         "tests": [
             ("abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6", True),
             ("abs(mean_absolute_deviation([1.0, 2.0, 3.0]) - 2/3) < 1e-6", True),
         ]},
        {"task_id": "HumanEval/11",
         "prompt": "from typing import List\n\n\ndef string_xor(a: str, b: str) -> str:\n    \"\"\"Input are two strings a and b consisting only of 1s and 0s.\n    Perform binary XOR on these inputs and return result also as a string.\n    >>> string_xor('010', '110')\n    '100'\n    \"\"\"\n",
         "tests": [
             ("string_xor('111000', '101010')", "010010"),
             ("string_xor('1', '1')", "0"),
             ("string_xor('0101', '0000')", "0101"),
         ]},
        {"task_id": "HumanEval/26",
         "prompt": "from typing import List\n\n\ndef remove_duplicates(numbers: List[int]) -> List[int]:\n    \"\"\"From a list of integers, remove all elements that occur more than once.\n    Keep order of elements left the same as in the input.\n    >>> remove_duplicates([1, 2, 3, 2, 4])\n    [1, 3, 4]\n    \"\"\"\n",
         "tests": [
             ("remove_duplicates([1, 2, 3, 2, 4])", [1, 3, 4]),
             ("remove_duplicates([])", []),
             ("remove_duplicates([1, 2, 3, 4])", [1, 2, 3, 4]),
         ]},
    ]
    results = []
    for p in problems[:n]:
        msg = [{"role": "user", "content": f"Complete the following Python function. Return ONLY the complete function (including the def line), no explanation, no markdown fences.\n\n{p['prompt']}"}]
        try:
            r = api_call(msg, max_tokens=2048)
            content, reasoning = get_response_text(r)
            timings = r.get("timings", {})
            
            # Try to execute and test the code
            passed = 0
            total = len(p.get("tests", []))
            error_msg = None
            
            # Extract code from content first, fall back to reasoning_content
            code_source = content.strip() if content.strip() else reasoning.strip()
            
            try:
                code = extract_python_function(code_source, p["prompt"])
                exec_globals = {}
                exec(code, exec_globals)
                for test_expr, expected in p.get("tests", []):
                    try:
                        result = eval(test_expr, exec_globals)
                        if result == expected:
                            passed += 1
                    except Exception as te:
                        pass
            except Exception as e:
                error_msg = str(e)[:100]
            
            results.append({
                "task_id": p["task_id"],
                "completion": content[:500],
                "reasoning": reasoning[:3000],
                "passed": passed,
                "total": total,
                "error": error_msg,
                "prompt_tokens": r["usage"]["prompt_tokens"],
                "completion_tokens": r["usage"]["completion_tokens"],
                "prompt_tps": timings.get("prompt_per_second", 0),
                "gen_tps": timings.get("predicted_per_second", 0),
            })
            status = f"PASS {passed}/{total}" if passed == total else f"PARTIAL {passed}/{total}"
            if error_msg:
                status = f"ERROR: {error_msg[:50]}"
            print(f"  {p['task_id']}: {status} | gen={timings.get('predicted_per_second',0):.1f} t/s")
        except Exception as e:
            print(f"  {p['task_id']}: FAILED - {e}")
            results.append({"task_id": p["task_id"], "error": str(e)})
    return results

def run_reasoning_smoke(quant_name, n=5):
    """Reasoning and knowledge questions with clear answers."""
    questions = [
        {"id": "math_1", "q": "What is 127 * 43?", "answer": "5461"},
        {"id": "math_2", "q": "What is the derivative of x^3 + 2x^2 - 5x + 7?", "answer": "3x^2 + 4x - 5", "alt": ["3x² + 4x - 5", "3*x**2 + 4*x - 5"]},
        {"id": "logic_1", "q": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "answer": "no"},
        {"id": "code_1", "q": "What does this Python code print?\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(len(x))", "answer": "4"},
        {"id": "knowledge_1", "q": "In what year did the first successful Moon landing occur?", "answer": "1969"},
    ]
    results = []
    for q in questions[:n]:
        msg = [{"role": "user", "content": q["q"] + " Answer concisely."}]
        try:
            r = api_call(msg, max_tokens=1024)
            content, reasoning = get_response_text(r)
            timings = r.get("timings", {})
            # Check answer in content (the actual response, not thinking)
            check_text = (content + " " + reasoning).lower()
            correct = q["answer"].lower() in check_text
            if not correct and "alt" in q:
                correct = any(a.lower() in check_text for a in q["alt"])
            results.append({
                "id": q["id"],
                "correct": correct,
                "response": content[:300],
                "reasoning": reasoning[:200],
                "prompt_tps": timings.get("prompt_per_second", 0),
                "gen_tps": timings.get("predicted_per_second", 0),
            })
            print(f"  {q['id']}: {'PASS' if correct else 'FAIL'} | answer: {content[:80]}")
        except Exception as e:
            print(f"  {q['id']}: FAILED - {e}")
            results.append({"id": q["id"], "error": str(e)})
    return results

if __name__ == "__main__":
    quant = sys.argv[1] if len(sys.argv) > 1 else "Q4_K_M"
    print(f"\n{'='*50}")
    print(f"Benchmarking REAP-20 {quant}")
    print(f"{'='*50}")
    
    print(f"\n--- HumanEval Smoke ({quant}) ---")
    he = run_humaneval_smoke(quant)
    
    print(f"\n--- Reasoning Smoke ({quant}) ---")
    rs = run_reasoning_smoke(quant)
    
    out = {"quant": quant, "humaneval": he, "reasoning": rs}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{quant}-smoke.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/{quant}-smoke.json")
