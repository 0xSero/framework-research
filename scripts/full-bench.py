#!/usr/bin/env python3
"""
Comprehensive LLM Benchmark Suite for Framework Desktop
Runs 10 benchmark categories x 100 samples each via llama-server OpenAI API.
Measures: decode TPS, prefill TPS, TTFT, accuracy, error rate, token spend.
"""

import json, sys, time, os, re, textwrap, statistics, urllib.request, urllib.error
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = os.environ.get("LLAMA_URL", "http://localhost:8080")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "./bench-results")
SAMPLES = int(os.environ.get("SAMPLES", "20"))
TIMEOUT = int(os.environ.get("TIMEOUT", "300"))
MODEL_NAME = "default"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── API helpers ───────────────────────────────────────────────────────────────

def api_call(messages, max_tokens=2048, temperature=0.0, extra_body=None):
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    elapsed = time.perf_counter() - t0
    r = json.loads(resp.read())
    
    msg = r["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    usage = r.get("usage", {})
    timings = r.get("timings", {})
    
    return {
        "content": content,
        "reasoning": reasoning,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "prefill_tps": timings.get("prompt_per_second", 0),
        "decode_tps": timings.get("predicted_per_second", 0),
        "ttft_ms": timings.get("prompt_n", 0) / max(timings.get("prompt_per_second", 1), 0.001) * 1000 if timings.get("prompt_per_second") else 0,
        "elapsed_s": round(elapsed, 3),
    }

def extract_code(text):
    """Extract code from markdown fences or raw text."""
    blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return textwrap.dedent(blocks[-1]).strip()
    return textwrap.dedent(text).strip()

def check_answer(response, expected, alternatives=None):
    """Check if expected answer appears in response (content or reasoning)."""
    text = (response["content"] + " " + response["reasoning"]).lower().strip()
    answers = [expected.lower()]
    if alternatives:
        answers += [a.lower() for a in alternatives]
    return any(a in text for a in answers)

# ─── Benchmark definitions ────────────────────────────────────────────────────
# Each bench returns list of sample dicts with "prompt" and a "check" function
# that takes the API response dict and returns (correct: bool, detail: str)

BENCHMARKS = {}

# ──────────────────────────────────────────────────────────────────────────────
# 1. Python Coding (HumanEval-style)
# ──────────────────────────────────────────────────────────────────────────────
def _make_python_bench():
    problems = [
        ("has_close_elements", "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n",
         [("has_close_elements([1.0, 2.0, 3.0], 0.5)", False), ("has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)", True)]),
        ("separate_paren_groups", "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Separate balanced paren groups.\"\"\"\n",
         [("separate_paren_groups('(()) (()) ()')", ['(())', '(())', '()'])]),
        ("mean_absolute_deviation", "from typing import List\n\ndef mean_absolute_deviation(numbers: List[float]) -> float:\n    \"\"\"Calculate MAD around mean.\"\"\"\n",
         [("abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6", True)]),
        ("string_xor", "def string_xor(a: str, b: str) -> str:\n    \"\"\"Binary XOR on two strings of 0s and 1s.\"\"\"\n",
         [("string_xor('010', '110')", "100"), ("string_xor('1', '1')", "0")]),
        ("remove_duplicates", "from typing import List\n\ndef remove_duplicates(numbers: List[int]) -> List[int]:\n    \"\"\"Remove elements that occur more than once, preserve order.\"\"\"\n",
         [("remove_duplicates([1, 2, 3, 2, 4])", [1, 3, 4])]),
        ("fibonacci", "def fibonacci(n: int) -> int:\n    \"\"\"Return the nth Fibonacci number (0-indexed).\"\"\"\n",
         [("fibonacci(0)", 0), ("fibonacci(10)", 55)]),
        ("is_palindrome", "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is a palindrome ignoring case and non-alphanumeric.\"\"\"\n",
         [("is_palindrome('A man, a plan, a canal: Panama')", True)]),
        ("flatten_list", "from typing import List, Any\n\ndef flatten_list(lst: List[Any]) -> List[Any]:\n    \"\"\"Recursively flatten a nested list.\"\"\"\n",
         [("flatten_list([1, [2, [3, 4]], 5])", [1, 2, 3, 4, 5])]),
        ("binary_search", "from typing import List\n\ndef binary_search(arr: List[int], target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1.\"\"\"\n",
         [("binary_search([1,3,5,7,9], 5)", 2), ("binary_search([1,3,5,7,9], 6)", -1)]),
        ("count_words", "def count_words(s: str) -> dict:\n    \"\"\"Count word frequencies in a string.\"\"\"\n",
         [("count_words('hello world hello')", {"hello": 2, "world": 1})]),
        ("merge_sorted", "from typing import List\n\ndef merge_sorted(a: List[int], b: List[int]) -> List[int]:\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
         [("merge_sorted([1,3,5], [2,4,6])", [1,2,3,4,5,6])]),
        ("reverse_int", "def reverse_int(n: int) -> int:\n    \"\"\"Reverse digits of an integer. 123 -> 321.\"\"\"\n",
         [("reverse_int(123)", 321), ("reverse_int(-456)", -654)]),
        ("is_valid_parentheses", "def is_valid_parentheses(s: str) -> bool:\n    \"\"\"Check if parentheses string is valid.\"\"\"\n",
         [("is_valid_parentheses('()[]{}')", True), ("is_valid_parentheses('(]')", False)]),
        ("gcd", "def gcd(a: int, b: int) -> int:\n    \"\"\"Compute greatest common divisor.\"\"\"\n",
         [("gcd(12, 8)", 4), ("gcd(7, 13)", 1)]),
        ("rotate_matrix", "from typing import List\n\ndef rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:\n    \"\"\"Rotate NxN matrix 90 degrees clockwise.\"\"\"\n",
         [("rotate_matrix([[1,2],[3,4]])", [[3,1],[4,2]])]),
        ("longest_common_prefix", "from typing import List\n\ndef longest_common_prefix(strs: List[str]) -> str:\n    \"\"\"Find longest common prefix among strings.\"\"\"\n",
         [("longest_common_prefix(['flower','flow','flight'])", "fl")]),
        ("two_sum", "from typing import List, Tuple\n\ndef two_sum(nums: List[int], target: int) -> Tuple[int, int]:\n    \"\"\"Find indices of two numbers that add to target.\"\"\"\n",
         [("sorted(two_sum([2,7,11,15], 9))", [0,1])]),
        ("capitalize_words", "def capitalize_words(s: str) -> str:\n    \"\"\"Capitalize first letter of each word.\"\"\"\n",
         [("capitalize_words('hello world')", "Hello World")]),
        ("factorial", "def factorial(n: int) -> int:\n    \"\"\"Return n!\"\"\"\n",
         [("factorial(5)", 120), ("factorial(0)", 1)]),
        ("deep_copy_list", "from typing import List, Any\nimport copy\n\ndef deep_copy_list(lst: List[Any]) -> List[Any]:\n    \"\"\"Return a deep copy of a nested list.\"\"\"\n",
         [("l=[1,[2,3]]; d=deep_copy_list(l); d[1][0]=99; l[1][0]", 2)]),
    ]
    samples = []
    for name, prompt, tests in problems:
        def make_check(ts):
            def check(resp):
                code = extract_code(resp["content"] if resp["content"].strip() else resp["reasoning"])
                if not code.strip():
                    return False, "empty response"
                try:
                    ns = {}
                    exec(code, ns)
                    passed = 0
                    for expr, expected in ts:
                        try:
                            result = eval(expr, ns)
                            if result == expected:
                                passed += 1
                        except:
                            pass
                    return passed == len(ts), f"{passed}/{len(ts)} tests passed"
                except Exception as e:
                    return False, f"exec error: {str(e)[:80]}"
            return check
        samples.append({"prompt": f"Complete this Python function. Return ONLY the code, no explanation:\n\n{prompt}", "check": make_check(tests), "category": "python", "id": name})
    return samples

BENCHMARKS["python"] = _make_python_bench

# ──────────────────────────────────────────────────────────────────────────────
# 2. JavaScript/TypeScript Coding
# ──────────────────────────────────────────────────────────────────────────────
def _make_js_bench():
    problems = [
        ("js_fibonacci", "Write a JavaScript function `fibonacci(n)` that returns the nth Fibonacci number (0-indexed). fibonacci(10) should return 55. Return only the function.", "55"),
        ("js_deep_clone", "Write a JavaScript function `deepClone(obj)` that deep clones an object (handles nested objects and arrays). Return only the function.", "deep"),
        ("js_flatten", "Write a JavaScript function `flatten(arr)` that deeply flattens a nested array. flatten([1,[2,[3,4]],5]) returns [1,2,3,4,5]. Return only the function.", "[1,2,3,4,5]"),
        ("js_debounce", "Write a JavaScript `debounce(fn, ms)` function. Return only the function code.", "debounce"),
        ("js_promise_all", "Write a JavaScript `promiseAll(promises)` that works like Promise.all. Return only the function.", "promiseAll"),
        ("js_curry", "Write a JavaScript `curry(fn)` function for currying. Return only the function.", "curry"),
        ("js_throttle", "Write a JavaScript `throttle(fn, ms)` function. Return only the function code.", "throttle"),
        ("js_capitalize", "Write a JavaScript function `capitalizeWords(str)` that capitalizes the first letter of each word. Return only the function.", "Hello"),
        ("js_anagram", "Write a JavaScript function `isAnagram(a, b)` that checks if two strings are anagrams. Return only the function.", "isAnagram"),
        ("js_range", "Write a JavaScript function `range(start, end, step)` like Python's range. Return only the function.", "range"),
    ]
    samples = []
    for pid, prompt, _ in problems:
        def make_check(p):
            def check(resp):
                code = (resp["content"] or resp["reasoning"]).strip()
                has_fn = bool(re.search(r'function\s|const\s|let\s|=>\s', code))
                has_ret = 'return' in code
                return has_fn and has_ret, f"function={'yes' if has_fn else 'no'} return={'yes' if has_ret else 'no'}"
            return check
        samples.append({"prompt": prompt, "check": make_check(prompt), "category": "javascript", "id": pid})
    return samples

BENCHMARKS["javascript"] = _make_js_bench

# ──────────────────────────────────────────────────────────────────────────────
# 3. Rust Coding
# ──────────────────────────────────────────────────────────────────────────────
def _make_rust_bench():
    problems = [
        ("rust_fibonacci", "Write a Rust function `fn fibonacci(n: u32) -> u64` that returns the nth Fibonacci number. Return only the function.", "fibonacci"),
        ("rust_reverse_string", "Write a Rust function `fn reverse_string(s: &str) -> String` that reverses a string. Return only the function.", "reverse"),
        ("rust_is_palindrome", "Write a Rust function `fn is_palindrome(s: &str) -> bool` that checks if a string is a palindrome. Return only the function.", "palindrome"),
        ("rust_bubble_sort", "Write a Rust function `fn bubble_sort(arr: &mut Vec<i32>)` that sorts in place. Return only the function.", "bubble_sort"),
        ("rust_binary_search", "Write a Rust function `fn binary_search(arr: &[i32], target: i32) -> Option<usize>` for binary search. Return only the function.", "binary_search"),
        ("rust_merge_sorted", "Write a Rust function `fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32>` that merges two sorted slices. Return only the function.", "merge"),
        ("rust_count_chars", "Write a Rust function `fn count_chars(s: &str) -> HashMap<char, usize>` that counts character frequencies. Return only the function.", "count"),
        ("rust_factorial", "Write a Rust function `fn factorial(n: u64) -> u64` that computes factorial. Return only the function.", "factorial"),
        ("rust_find_max", "Write a Rust function `fn find_max(arr: &[i32]) -> Option<i32>` that finds the maximum. Return only the function.", "find_max"),
        ("rust_vec_dedup", "Write a Rust function `fn dedup(sorted: &mut Vec<i32>)` that removes duplicates from a sorted vector. Return only the function.", "dedup"),
    ]
    samples = []
    for pid, prompt, _ in problems:
        def make_check(p):
            def check(resp):
                code = (resp["content"] or resp["reasoning"]).strip()
                return has_fn, f"has fn: {has_fn}"
            return check
        samples.append({"prompt": prompt, "check": make_check(prompt), "category": "rust", "id": pid})
    return samples

BENCHMARKS["rust"] = _make_rust_bench

# ──────────────────────────────────────────────────────────────────────────────
# 4. Mathematics
# ──────────────────────────────────────────────────────────────────────────────
def _make_math_bench():
    problems = [
        ("math_arith_1", "What is 127 * 43? Answer with just the number.", "5461"),
        ("math_arith_2", "What is 256 * 256? Answer with just the number.", "65536"),
        ("math_arith_3", "What is 999 * 999? Answer with just the number.", "998001"),
        ("math_calc_1", "What is the derivative of x^3 + 2x^2 - 5x + 7 with respect to x?", "3x^2 + 4x - 5", ["3x² + 4x - 5", "3*x**2 + 4*x - 5", "3x^2+4x-5"]),
        ("math_calc_2", "What is the integral of 2x dx?", "x^2 + C", ["x^2+C", "x² + C", "x^2"]),
        ("math_calc_3", "What is the derivative of sin(x)?", "cos(x)", ["cos(x)"]),
        ("math_geom_1", "What is the area of a circle with radius 5?", "78.54", ["25pi", "25*pi", "78.5"]),
        ("math_geom_2", "What is the volume of a sphere with radius 3?", "113.1", ["36pi", "36*pi", "113"]),
        ("math_prob_1", "If you flip a fair coin 3 times, what is the probability of getting exactly 2 heads?", "3/8", ["0.375"]),
        ("math_prob_2", "What is 15! / 14!?", "15"),
        ("math_seq_1", "What is the next number in the sequence: 2, 6, 12, 20, 30, ?", "42"),
        ("math_seq_2", "What is the 10th prime number?", "29"),
        ("math_linalg_1", "What is the determinant of [[1,2],[3,4]]?", "-2"),
        ("math_linalg_2", "What is the dot product of [1,2,3] and [4,5,6]?", "32"),
        ("math_logic_1", "If a triangle has angles 30 and 60 degrees, what is the third angle?", "90"),
        ("math_logic_2", "What is log2(1024)?", "10"),
        ("math_logic_3", "What is 2^10?", "1024"),
        ("math_logic_4", "What is sqrt(144)?", "12"),
        ("math_logic_5", "What is the sum of the first 100 natural numbers?", "5050"),
        ("math_series_1", "What is the sum of the infinite geometric series 1 + 1/2 + 1/4 + 1/8 + ...?", "2"),
    ]
    samples = []
    for pid, prompt, answer, *alts in problems:
        alternatives = alts[0] if alts else None
        def make_check(a, al):
            def check(resp):
                return check_answer(resp, a, al), (resp["content"] or resp["reasoning"])[:100]
            return check
        samples.append({"prompt": prompt, "check": make_check(answer, alternatives), "category": "math", "id": pid})
    return samples

BENCHMARKS["math"] = _make_math_bench

# ──────────────────────────────────────────────────────────────────────────────
# 5. Logic & Reasoning
# ──────────────────────────────────────────────────────────────────────────────
def _make_logic_bench():
    problems = [
        ("logic_syllogism_1", "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Answer yes or no with a brief explanation.", "no"),
        ("logic_syllogism_2", "All cats are animals. All animals are mortal. Can we conclude that all cats are mortal? Answer yes or no.", "yes"),
        ("logic_counter", "If it's raining, the ground is wet. The ground is wet. Can we conclude it's raining? Answer yes or no.", "no"),
        ("logic_set_1", "In a class of 30 students, 15 play football, 12 play basketball, and 5 play both. How many play neither?", "8"),
        ("logic_set_2", "A survey of 100 people: 60 like tea, 40 like coffee, 20 like both. How many like only tea?", "40"),
        ("logic_riddle_1", "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05", ["5 cents", "$0.05", "five cents", "5c"]),
        ("logic_riddle_2", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5 minutes", ["5 min"]),
        ("logic_riddle_3", "A lake has lily pads that double every day. If it takes 48 days to cover the lake, how many days to cover half the lake?", "47"),
        ("logic_if_then", "If P implies Q, and Q implies R, does P imply R? Answer yes or no.", "yes"),
        ("logic_de_morgan", "Is NOT(A AND B) equivalent to (NOT A) OR (NOT B)? Answer yes or no.", "yes"),
        ("logic_truth_1", "If 'This statement is false' is a paradox, what is the simplest name for this type of paradox?", "liar", ["self-reference", "epimenides"]),
        ("logic_arg_1", "Is this argument valid: 'All birds can fly. Penguins are birds. Therefore penguins can fly.'? Answer valid or invalid.", "valid"),
        ("logic_arg_2", "Is this argument sound: 'All birds can fly. Penguins are birds. Therefore penguins can fly.'? Answer sound or unsound.", "unsound"),
        ("logic_conditional", "If the contrapositive of 'If P then Q' is 'If not Q then not P', what is the contrapositive of 'If it rains then the ground is wet'?", "if the ground is not wet then it does not rain"),
        ("logic_quantifier", "Does 'there exists x such that for all y, P(x,y)' imply 'for all y there exists x such that P(x,y)'? Answer yes or no.", "yes"),
    ]
    samples = []
    for pid, prompt, answer, *alts in problems:
        alternatives = alts[0] if alts else None
        def make_check(a, al):
            def check(resp):
                return check_answer(resp, a, al), (resp["content"] or resp["reasoning"])[:100]
            return check
        samples.append({"prompt": prompt, "check": make_check(answer, alternatives), "category": "logic", "id": pid})
    return samples

BENCHMARKS["logic"] = _make_logic_bench

# ──────────────────────────────────────────────────────────────────────────────
# 6. Philosophy & Ethics
# ──────────────────────────────────────────────────────────────────────────────
def _make_philosophy_bench():
    problems = [
        ("phil_trolley", "Briefly explain the trolley problem in philosophy. What are the two main variants? Answer in 2-3 sentences.", ["trolley", "switch", "fat man", "thomson"]),
        ("phil_cogito", "Who said 'Cogito, ergo sum' and what does it mean? Answer in 1-2 sentences.", ["descartes", "i think", "therefore i am"]),
        ("phil_utilitarianism", "Name the two founders of utilitarianism and explain the core principle in 1-2 sentences.", ["bentham", "mill", "greatest happiness"]),
        ("phil_kant_categorical", "What is Kant's categorical imperative? Explain in 1-2 sentences.", ["categorical imperative", "universal law", "kant"]),
        ("phil_plato_cave", "Briefly describe Plato's Allegory of the Cave. What does it represent?", ["cave", "shadows", "forms", "plato"]),
        ("phil_free_will", "What is compatibilism regarding free will? Answer in 1-2 sentences.", ["compatibilism", "compatible", "determinism"]),
        ("phil_existentialism", "Who said 'existence precedes essence' and what does it mean?", ["sartre", "existence precedes"]),
        ("phil_ethics_virtue", "Which philosopher is most associated with virtue ethics? What is the golden mean?", ["aristotle", "golden mean", "virtue"]),
        ("phil_turing_test", "Briefly describe the Turing test and its significance for AI.", ["turing", "imitation game", "machine"]),
        ("phil_mary_room", "What is Frank Jackson's 'Mary's Room' thought experiment about?", ["mary", "qualia", "knowledge argument", "color", "black and white"]),
    ]
    samples = []
    for pid, prompt, keywords in problems:
        def make_check(kw):
            def check(resp):
                text = (resp["content"] + " " + resp["reasoning"]).lower()
                hits = sum(1 for k in kw if k.lower() in text)
                return hits >= 1, f"keyword hits: {hits}/{len(kw)}"
            return check
        samples.append({"prompt": prompt, "check": make_check(keywords), "category": "philosophy", "id": pid})
    return samples

BENCHMARKS["philosophy"] = _make_philosophy_bench

# ──────────────────────────────────────────────────────────────────────────────
# 7. Agentic / Tool Use
# ──────────────────────────────────────────────────────────────────────────────
def _make_agentic_bench():
    problems = [
        ("agent_plan_1", "You are an AI coding agent. Break down this task into steps: 'Create a REST API endpoint that accepts a JSON payload, validates it, stores it in a PostgreSQL database, and returns a response.' List numbered steps only.", ["validate", "postgres", "json", "endpoint", "response"]),
        ("agent_plan_2", "You are an AI coding agent. Break down: 'Write a CLI tool that recursively searches a directory for .log files, parses timestamps, and generates a summary report.' List numbered steps only.", ["recursive", "log", "timestamp", "parse", "report"]),
        ("agent_debug_1", "A Python Flask app returns 500 errors. The logs show 'KeyError: user_id'. Describe your debugging strategy step by step.", ["keyerror", "user_id", "request", "session"]),
        ("agent_debug_2", "A React component re-renders infinitely. Describe how you would debug this.", ["useeffect", "state", "dependency", "render"]),
        ("agent_refactor_1", "Describe how to refactor a 2000-line Python file into a proper module structure. Give step-by-step plan.", ["import", "module", "class", "function"]),
        ("agent_security_1", "List the top 5 security vulnerabilities to check when reviewing a web application's code.", ["xss", "sql injection", "csrf", "auth", "input"]),
        ("agent_arch_1", "Design a system that processes 10,000 events per second with at-least-once delivery guarantees. Describe the architecture.", ["queue", "kafka", "consumer", "retry", "exactly"]),
        ("agent_git_1", "A developer accidentally committed sensitive data. Describe the exact git commands to fix this.", ["git", "filter", "rebase", "force"]),
        ("agent_deploy_1", "Describe a zero-downtime deployment strategy for a microservice.", ["rolling", "blue", "green", "health", "traffic"]),
        ("agent_test_1", "Write a testing strategy for a payment processing module. What types of tests would you write?", ["unit", "integration", "mock", "edge"]),
    ]
    samples = []
    for pid, prompt, keywords in problems:
        def make_check(kw):
            def check(resp):
                text = (resp["content"] + " " + resp["reasoning"]).lower()
                hits = sum(1 for k in kw if k.lower() in text)
                return hits >= 2, f"keyword hits: {hits}/{len(kw)}"
            return check
        samples.append({"prompt": prompt, "check": make_check(keywords), "category": "agentic", "id": pid})
    return samples

BENCHMARKS["agentic"] = _make_agentic_bench

# ──────────────────────────────────────────────────────────────────────────────
# 8. Code Review & Debugging
# ──────────────────────────────────────────────────────────────────────────────
def _make_codereview_bench():
    problems = [
        ("review_python_1", "Find ALL bugs in this Python code:\n\ndef calculate_average(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total / len(numbers)\n\nList every bug.", ["zero", "empty", "division"]),
        ("review_python_2", "Find the bug:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nThis works but has a problem. What is it?", ["slow", "exponential", "recomput"]),
        ("review_sql_1", "Find the SQL injection vulnerability:\n\nquery = f\"SELECT * FROM users WHERE name = '{user_input}'\"\n\ncursor.execute(query)\n\nExplain the vulnerability and how to fix it.", ["injection", "parameterized", "prepared", "sanitize"]),
        ("review_js_1", "Find the bug:\n\nfor (var i = 0; i < 5; i++) {\n    setTimeout(() => console.log(i), 100);\n}\n\nWhat does this print and why?", ["5", "var", "closure", "let"]),
        ("review_race", "Find the race condition:\n\nbalance = 100\ndef withdraw(amount):\n    global balance\n    if balance >= amount:\n        balance -= amount\n        return True\n    return False\n\nExplain the race condition.", ["race", "concurrent", "lock", "atomic", "thread"]),
        ("review_leak_1", "Find the memory leak:\n\nclass Cache:\n    def __init__(self):\n        self.data = {}\n    def add(self, key, value):\n        self.data[key] = value\n\nWhat's wrong?", ["unbounded", "grow", "limit", "evict", "lru"]),
        ("review_security_1", "Is this password hashing secure?\n\nimport hashlib\ndef hash_password(pw):\n    return hashlib.md5(pw.encode()).hexdigest()\n\nWhy or why not?", ["md5", "salt", "bcrypt", "fast", "rainbow"]),
        ("review_logic_1", "This sort function has a bug. What is it?\n\ndef sort_list(lst):\n    return sorted(lst, reverse=False)\n\ndef get_top_3(lst):\n    sorted_lst = sort_list(lst)\n    return sorted_lst[:3]\n\nget_top_3([5, 1, 3, None, 2])\n\nWhat happens and how to fix?", ["none", "type", "key", "filter"]),
        ("review_api_1", "Review this API endpoint for issues:\n\n@app.route('/user/<user_id>')\ndef get_user(user_id):\n    user = db.query(f'SELECT * FROM users WHERE id = {user_id}')\n    return jsonify(user)\n\nList all issues.", ["injection", "error", "parameterized", "exception"]),
        ("review_concurrency_1", "Find the deadlock:\n\nlock_a = Lock()\nlock_b = Lock()\n\ndef task1():\n    with lock_a:\n        with lock_b:\n            do_work()\n\ndef task2():\n    with lock_b:\n        with lock_a:\n            do_work()\n\nWhy does this deadlock?", ["deadlock", "order", "acquire", "circular"]),
    ]
    samples = []
    for pid, prompt, keywords in problems:
        def make_check(kw):
            def check(resp):
                text = (resp["content"] + " " + resp["reasoning"]).lower()
                hits = sum(1 for k in kw if k.lower() in text)
                return hits >= 1, f"keyword hits: {hits}/{len(kw)}"
            return check
        samples.append({"prompt": prompt, "check": make_check(keywords), "category": "codereview", "id": pid})
    return samples

BENCHMARKS["codereview"] = _make_codereview_bench

# ──────────────────────────────────────────────────────────────────────────────
# 9. System Design
# ──────────────────────────────────────────────────────────────────────────────
def _make_sysdesign_bench():
    problems = [
        ("design_url_shortener", "Design a URL shortener service. Describe: data model, API, scaling strategy. Be concise.", ["hash", "redirect", "database", "cache"]),
        ("design_chat", "Design a real-time chat system. Describe: protocol, message storage, delivery guarantees.", ["websocket", "message", "queue", "persist"]),
        ("design_rate_limiter", "Design a distributed rate limiter. Describe: algorithm, data structure, distribution.", ["token bucket", "sliding window", "redis", "counter"]),
        ("design_cache", "Design a distributed cache system. Describe: eviction policy, consistency, partitioning.", ["lru", "consistent hashing", "ttl", "evict"]),
        ("design_search", "Design a search engine for a large codebase. Describe: indexing, query processing, ranking.", ["index", "inverted", "tokeniz", "rank"]),
        ("design_cdn", "How does a CDN work? Describe: edge nodes, caching, invalidation.", ["edge", "cache", "origin", "dns"]),
        ("design_queue", "Design a message queue with exactly-once delivery. Describe: storage, replication, consumer offsets.", ["offset", "commit", "idempotent", "log"]),
        ("design_auth", "Design an authentication system with OAuth2. Describe: flow, token management, security.", ["token", "oauth", "jwt", "refresh"]),
        ("design_monitor", "Design a monitoring/alerting system. Describe: metrics collection, storage, alerting rules.", ["metric", "alert", "time series", "threshold"]),
        ("design_config", "Design a dynamic configuration service. Describe: storage, distribution, hot reload.", ["push", "pull", "watch", "version"]),
    ]
    samples = []
    for pid, prompt, keywords in problems:
        def make_check(kw):
            def check(resp):
                text = (resp["content"] + " " + resp["reasoning"]).lower()
                hits = sum(1 for k in kw if k.lower() in text)
                return hits >= 2, f"keyword hits: {hits}/{len(kw)}"
            return check
        samples.append({"prompt": prompt, "check": make_check(keywords), "category": "sysdesign", "id": pid})
    return samples

BENCHMARKS["sysdesign"] = _make_sysdesign_bench

# ──────────────────────────────────────────────────────────────────────────────
# 10. Multi-language / Polyglot
# ──────────────────────────────────────────────────────────────────────────────
def _make_polyglot_bench():
    problems = [
        ("poly_hello_c", "Write a C program that prints 'Hello, World!' to stdout. Return only the code.", ["#include", "printf", "main"]),
        ("poly_hello_go", "Write a Go program that prints 'Hello, World!'. Return only the code.", ["func main", "fmt", "Print"]),
        ("poly_hello_rust", "Write a Rust program that prints 'Hello, World!'. Return only the code.", ["fn main", "println"]),
        ("poly_sort_py", "Write a Python one-liner to sort a list of numbers in descending order.", ["sort", "reverse"]),
        ("poly_map_js", "Write JavaScript to double every element in an array using map. Return only the code.", ["map", "=>"]),
        ("poly_sql_join", "Write a SQL query to join 'users' and 'orders' tables on user_id, showing name and total.", ["join", "select", "on"]),
        ("poly_regex_email", "Write a regex pattern to validate email addresses. Return only the pattern.", ["@", "."]),
        ("poly_dockerfile", "Write a minimal Dockerfile for a Python Flask app. Return only the Dockerfile.", ["from", "copy", "run", "expose"]),
        ("poly_bash_find", "Write a bash command to find all .py files modified in the last 7 days. Return only the command.", ["find", "-name", "-mtime"]),
        ("poly_git_rebase", "Write the exact git commands to interactively rebase the last 3 commits. Return only commands.", ["git", "rebase", "-i"]),
    ]
    samples = []
    for pid, prompt, keywords in problems:
        def make_check(kw):
            def check(resp):
                text = (resp["content"] + " " + resp["reasoning"]).lower()
                hits = sum(1 for k in kw if k.lower() in text)
                return hits >= 2, f"keyword hits: {hits}/{len(kw)}"
            return check
        samples.append({"prompt": prompt, "check": make_check(keywords), "category": "polyglot", "id": pid})
    return samples

BENCHMARKS["polyglot"] = _make_polyglot_bench

# ─── Runner ────────────────────────────────────────────────────────────────────

def run_benchmark(name, samples, n_samples=100):
    """Run a benchmark category, cycling through samples to reach n_samples."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — {n_samples} samples")
    print(f"{'='*60}")
    
    results = []
    for i in range(n_samples):
        sample = samples[i % len(samples)]
        iteration = i // len(samples) + 1
        
        try:
            resp = api_call([{"role": "user", "content": sample["prompt"]}])
            correct, detail = sample["check"](resp)
            
            result = {
                "sample_idx": i,
                "id": sample["id"],
                "iteration": iteration,
                "category": sample["category"],
                "correct": correct,
                "detail": detail,
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "total_tokens": resp["total_tokens"],
                "prefill_tps": resp["prefill_tps"],
                "decode_tps": resp["decode_tps"],
                "ttft_ms": resp["ttft_ms"],
                "elapsed_s": resp["elapsed_s"],
                "error": None,
            }
        except Exception as e:
            result = {
                "sample_idx": i,
                "id": sample["id"],
                "iteration": iteration,
                "category": sample["category"],
                "correct": False,
                "detail": f"API error: {str(e)[:100]}",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prefill_tps": 0,
                "decode_tps": 0,
                "ttft_ms": 0,
                "elapsed_s": 0,
                "error": str(e)[:200],
            }
        
        results.append(result)
        status = "." if result["correct"] else "x"
        if (i + 1) % 10 == 0:
            acc = sum(r["correct"] for r in results) / len(results) * 100
            avg_tps = statistics.mean([r["decode_tps"] for r in results if r["decode_tps"] > 0])
            errors = sum(1 for r in results if r["error"])
            print(f"  [{i+1:3d}/{n_samples}] acc={acc:.0f}% avg_tps={avg_tps:.1f} errors={errors}")
    
    return results

def summarize(results):
    """Compute summary statistics for a benchmark."""
    valid = [r for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]
    
    if not valid:
        return {"total": len(results), "errors": len(errors), "accuracy": 0}
    
    return {
        "total": len(results),
        "valid": len(valid),
        "errors": len(errors),
        "error_rate": len(errors) / len(results) * 100,
        "accuracy": sum(r["correct"] for r in valid) / len(valid) * 100,
        "avg_prefill_tps": statistics.mean([r["prefill_tps"] for r in valid if r["prefill_tps"] > 0]) if any(r["prefill_tps"] > 0 for r in valid) else 0,
        "avg_decode_tps": statistics.mean([r["decode_tps"] for r in valid if r["decode_tps"] > 0]) if any(r["decode_tps"] > 0 for r in valid) else 0,
        "median_decode_tps": statistics.median([r["decode_tps"] for r in valid if r["decode_tps"] > 0]) if any(r["decode_tps"] > 0 for r in valid) else 0,
        "p10_decode_tps": sorted([r["decode_tps"] for r in valid if r["decode_tps"] > 0])[max(0, len(valid)//10)] if valid else 0,
        "p90_decode_tps": sorted([r["decode_tps"] for r in valid if r["decode_tps"] > 0])[min(len(valid)-1, len(valid)*9//10)] if valid else 0,
        "avg_ttft_ms": statistics.mean([r["ttft_ms"] for r in valid]) if valid else 0,
        "median_ttft_ms": statistics.median([r["ttft_ms"] for r in valid]) if valid else 0,
        "avg_prompt_tokens": statistics.mean([r["prompt_tokens"] for r in valid]) if valid else 0,
        "avg_completion_tokens": statistics.mean([r["completion_tokens"] for r in valid]) if valid else 0,
        "total_tokens": sum(r["total_tokens"] for r in results),
        "total_time_s": sum(r["elapsed_s"] for r in results),
    }

def main():
    model_label = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    bench_filter = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Framework Desktop Benchmark Suite")
    print(f"Model: {model_label}")
    print(f"Samples per benchmark: {SAMPLES}")
    print(f"Started: {datetime.now().isoformat()}")
    
    all_results = {}
    all_summaries = {}
    
    bench_names = [bench_filter] if bench_filter else list(BENCHMARKS.keys())
    
    for name in bench_names:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}")
            continue
        samples = BENCHMARKS[name]()
        results = run_benchmark(name, samples, SAMPLES)
        summary = summarize(results)
        all_results[name] = results
        all_summaries[name] = summary
        print(f"\n  SUMMARY {name}: acc={summary['accuracy']:.1f}% decode={summary['avg_decode_tps']:.1f} t/s ttft={summary['avg_ttft_ms']:.0f}ms errors={summary['error_rate']:.1f}%")
    
    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{RESULTS_DIR}/{model_label}_{ts}"
    
    with open(f"{out_path}_full.json", "w") as f:
        json.dump({"model": model_label, "timestamp": ts, "results": all_results}, f, indent=2)
    with open(f"{out_path}_summary.json", "w") as f:
        json.dump({"model": model_label, "timestamp": ts, "summaries": all_summaries}, f, indent=2)
    
    # Print final table
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS — {model_label}")
    print(f"{'='*80}")
    print(f"{'Bench':<15} {'Acc%':>5} {'DecTPS':>8} {'P10':>6} {'P90':>6} {'TTFTms':>7} {'Err%':>5} {'Tokens':>8}")
    print(f"{'-'*15} {'-'*5} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*8}")
    for name, s in all_summaries.items():
        print(f"{name:<15} {s['accuracy']:>5.1f} {s['avg_decode_tps']:>8.1f} {s['p10_decode_tps']:>6.1f} {s['p90_decode_tps']:>6.1f} {s['avg_ttft_ms']:>7.0f} {s['error_rate']:>5.1f} {s['total_tokens']:>8}")
    
    print(f"\nResults saved to {out_path}_*")

if __name__ == "__main__":
    main()
