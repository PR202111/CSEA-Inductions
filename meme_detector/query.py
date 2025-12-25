import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from transformers import pipeline
from collect_data import get_meme_collection
from langchain_ollama import ChatOllama
import json


llm = ChatOllama(
    model='llama3.2',
    temperature=0.3
    )

collection = get_meme_collection()

ocr = PaddleOCR(use_textline_orientation=True, lang="en")

vibe_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=1
)


def process_meme(img_array):
   
    ocr_result = ocr.predict(img_array)
    extracted_words = []
    extracted_text = ""
    for line in ocr_result:
        if "rec_texts" in line:
            extracted_words.extend(line['rec_texts'])
    
    extracted_text = " ".join(extracted_words)

    if extracted_text.strip():
        bert_res = vibe_classifier(extracted_text)[0]
        print(bert_res)
        vibe_label = bert_res[0]["label"]
        label_confidence = bert_res[0]["score"]
    else:
        vibe_label = "neutral"
        label_confidence = 0

    results = collection.query(
        query_images=[img_array],   
        n_results=1
    )

    meta = results["metadatas"][0][0]

    
    return {
        "text": extracted_text,
        "detected_template": meta["title"],
        "current_vibe": vibe_label,
        "label_confidence": label_confidence
    }




def process_query(metadata:dict)->str:
    prompt = f'''
    you are a meme writer

    Original Meme Text: {metadata.get('text')}

    Detected Meme Template: {metadata.get('detected_template')}

    Detected Vibe or tone: {metadata.get('current_vibe')}

    Rewrite the meme text in the following styles:
    1. Sarcastic
    2. Wholesome
    3. Gen Z / brainrot humor

    Rules:
    - Keep each rewrite under 15 words
    - Do not explain
    - Output JSON only

    Format:
    {{
    "sarcastic": "rewriten in sarcastic way",
    "wholesome": "rewriten in wholesome way",
    "brainrot": "rewriten in brairot/gen-Z way"
    }}
'''
    result = llm.invoke(prompt)
    
    return json.loads(result.content)


