import os
import requests
import pandas as pd
import chromadb
import numpy as np

from tqdm import tqdm
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

CSV_PATH = "hf://datasets/sergiogpinto/memefact-templates/imkg_final_final_final_processor.csv"
IMAGE_DIR = "meme_templates"
CHROMA_DB_DIR = "meme_vector_db"

os.makedirs(IMAGE_DIR, exist_ok=True)


def download_templates(df):
    paths = []
    print("Downloading images...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row["template_url"]

        safe_title = (
            str(row["template_title"])[:20]
            .replace(" ", "_")
            .replace("/", "")
        )
        filename = f"{row['template_id']}_{safe_title}.jpg"
        path = os.path.join(IMAGE_DIR, filename)

        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(r.content)
                    paths.append(path)
                else:
                    paths.append(None)
            except:
                paths.append(None)
        else:
            paths.append(path)

    return paths



def get_meme_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    collection = client.get_or_create_collection(
        name="meme_templates",
        embedding_function=OpenCLIPEmbeddingFunction()
    )

    if collection.count() > 0:
        return collection


    df = pd.read_csv(CSV_PATH)
    df["local_path"] = download_templates(df)
    df = df.dropna(subset=["local_path"]).reset_index(drop=True)

    ids = []
    images = []
    metadatas = []


    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = Image.open(row["local_path"]).convert("RGB")
            img_array = np.array(img)

            images.append(img_array)
            ids.append(str(row["template_id"]))

            metadatas.append({
                "title": str(row["template_title"]),
                "about": str(row["about"]) if pd.notna(row["about"]) else "No info",
                "description": str(row["description"]),
                "vibe": str(row["caption_style_explanation"])
            })

        except Exception as e:
            print(f"Failed to load image {row['local_path']}: {e}")

  
    collection.upsert(
        ids=ids,
        images=images,        
        metadatas=metadatas
    )

    
    return collection
