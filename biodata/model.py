import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json
from dotenv import load_dotenv
import os
import re
load_dotenv()



def extract_json(text: str):
    """
    Extracts the first JSON object from an LLM response safely.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group())

llm = ChatOpenAI(
            model="gpt-4o", 
            api_key=os.getenv("OPEN_AI_API_KEY"),
            base_url=os.getenv("OPEN_AI_API_BASE"),
        )


# llm = ChatOllama(model='llama3.2',temperature=0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "final_flag_model")



tokenizer = DistilBertTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)
model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True
)



device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()  

def analyze_bio(text):
    labels = ["narcissistic","toxic","cringe","healthy"]
    

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    prompt = f"""
You are a dating profile safety and quality analyzer.

Your job is to convert psychological trait scores and bio text
into relationship "flags":

- Green flags = healthy, respectful, emotionally mature signals
- Yellow flags = mild concerns, sarcasm, immaturity, uncertainty
- Red flags = toxic, narcissistic, controlling, misogynistic, hostile signals

You will be given:
1) A dating bio
2) Model-predicted trait scores between 0 and 1:
- narcissistic
- toxic
- cringe
- healthy

You MUST:
- Decide which flags apply (green / yellow / red)
- Extract exact words or phrases from the bio that triggered each flag
- Briefly explain each flag in one short sentence
- Output JSON ONLY

Rules:
- healthy > 0.7 → green flags
- narcissistic or toxic > 0.6 → red flags
- cringe alone → yellow
- Do NOT invent words

---

Bio:
"{text}"

Scores:
healthy: {round(float(probs[3]), 3)}
narcissistic: {round(float(probs[0]), 3)}
toxic: {round(float(probs[1]), 3)}
cringe: {round(float(probs[2]), 3)}

Return only JSON in this format:
{{
  "green_flags": [
    {{
      "reason": "",
      "phrases": []
    }}
  ],
  "yellow_flags": [
    {{
      "reason": "",
      "phrases": []
    }}
  ],
  "red_flags": [
    {{
      "reason": "",
      "phrases": []
    }}
  ]
}}
"""

    response = llm.invoke([
        HumanMessage(content=prompt)
    ])
    

    return extract_json(response.content),dict(zip(labels, probs))

def llm_rewrite(text,tone,flags):
    prompt = f"""
You are a professional dating profile copywriter.

Your job is to rewrite dating bios to improve tone, safety, and attractiveness
while keeping the person's core personality intact.

You will be given:
1) An original dating bio
2) Detected flags (green / yellow / red) with triggering phrases
3) A target tone

Your task:
- Remove or rephrase any red or yellow flag language
- Preserve positive traits and intent
- Rewrite the bio to match the target tone
- Do NOT add new traits or claims
- Do NOT moralize or explain
- Output ONLY the rewritten bio (no markdown, no JSON, no commentary)

---

### TARGET TONE
{tone}

Examples of tones:
- healthy & respectful
- confident but humble
- playful & witty
- emotionally mature
- calm & grounded
- light-hearted

---

### ORIGINAL BIO
{text}

---

### DETECTED FLAGS
Red flags:
{flags['red_flags']}

Yellow flags:
{flags['yellow_flags']}

Green flags:
{flags['green_flags']}

---

### REWRITE RULES
- If a sentence causes a red flag → remove or neutralize it
- If a sentence causes a yellow flag → soften it
- Keep length similar to the original
- Keep first-person voice
- Avoid generic phrases like "I love travel and food"

---

### OUTPUT
Rewrite the bio now.
"""
    
    result = llm.invoke(prompt)
    print(result.content)

    return result.content



