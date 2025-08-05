import requests
import re 

def format_prompt(text):
    print("Kicking off Summary Job...")
    prompt = f"""
        You are reviewing a private equity firm's website description to identify its explicitly stated investment areas.

        Input text:
        {text}

        Your task is to extract only the information that is clearly and explicitly stated. Do **not** infer or assume investment focus based on context, vague language, employee backgrounds, or general descriptions. If the industry or thesis is not **directly stated** in the text, you must omit it.

        Instructions:
        1. Identify and list only the **explicitly stated industry focus areas**—industries the firm directly says it invests in.
        2. For each industry, extract any **specific investment thesis areas** (e.g., types of businesses, value creation strategies, market conditions, subsectors) that are **directly mentioned**.

        Important:
        - Ignore implications, suggestions, or inferred meanings.
        - Do not include areas just because they appear in example deals or portfolios unless the firm explicitly states them as a focus.
        - Do not include boilerplate descriptions or general language unless tied to a named industry.

        Format your response exactly like this:
        Industry Focus Areas and Specific Theses:
        - [Industry 1]
        - Thesis Area(s): 
            - [Bullet point]
            - [Bullet point]
        - [Industry 2]
        - Thesis Area(s): 
            - [Bullet point]
            - [Bullet point]
        (...continue for all explicitly stated industries)

        Example:
        If the input text includes a statement like: "We focus on value-added distribution businesses, particularly those providing essential infrastructure products," then your output should be:

        Industry Focus Areas and Specific Theses:
        - Value-Added Distribution
        - Thesis Area(s):
            - Providers of essential infrastructure products (e.g., waterworks suppliers like Core & Main)

        """
    return prompt

def format_grade_prompt(answer):
    print("Kicking off grade job")
    return f"""
        You are an evaluator checking whether the text below provides meaningful information about a private equity firm's investment thesis.

        Instructions:
        - Return **-1** if the text contains little or no explicit investment thesis information.
        - Return **1** if it contains clear, specific, and meaningful thesis information.
        - Do NOT explain your answer. Respond ONLY with -1 or 1.

        Examples:
        "Focused on acquiring founder-led SaaS companies in healthcare." → 1  
        "We are a values-based firm driven by partnerships." → -1

        Evaluate this:
        \"\"\"
        {answer}
        \"\"\"

        Your response:
        Only output -1 or 1.
        """


def call_model(text):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral:7b-instruct",
            "prompt": text,
            "stream": False
        }
    )
    return response.json()["response"]

def strip_thoughts(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()

def extract_first_int(text):
    match = re.search(r'\b-?\d+\b', text)
    if not match:
        raise ValueError(f"No integer found in response: {text}")
    return int(match.group())