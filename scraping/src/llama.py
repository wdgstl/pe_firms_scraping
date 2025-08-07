import requests
import re 

def format_prompt(text):
    print("Starting clean industry extraction…")
    prompt = f"""
Extract only the **explicitly stated** industry focus areas from the text below.

### INSTRUCTIONS:
- Return only the names of industries that are clearly and directly mentioned.
- Do NOT guess or infer. If it's not plainly stated, skip it.
- Do NOT include commentary, explanations, or implied references.
- Do NOT write anything other than clean industry names in brackets.

### OUTPUT FORMAT:
- Each line: [Industry Name]
- If there are no valid industries, return an empty string.
- Do NOT include lines like:
  [No industry stated]
  [Healthcare (implied)]
  [Not explicitly mentioned]

### EXAMPLE OUTPUT:
[Healthcare Services]  
[Value-Added Distribution]  
[Industrial Services]

### TEXT:
{text}
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


import re

def extract_industries(model_output: str) -> list[str]:
    industries = []
    matches = re.findall(r"\[([^\[\]\n]+)\]", model_output)

    for match in matches:
        lower = match.lower()
        if any(phrase in lower for phrase in [
            "no industry", "not stated", "implied", "not explicitly", "comment", "example", "private equity"
        ]):
            continue
        industries.append(match.strip())

    return industries
