import requests


def format_prompt(text):
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
        """
    
    return prompt

def call_mixtral(text):
    print("Kicking off Mixtral Job...")
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mixtral",
            "prompt": format_prompt(text),
            "stream": False
        }
    )
    return response.json()["response"]