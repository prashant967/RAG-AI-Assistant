import requests
import os
import json
import pandas as pd
import joblib

BATCH_SIZE = 200
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    if r.status_code != 200:
        print("Error:", r.text)
        return []
    return r.json()["embeddings"]

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0
for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"Creating Embeddings for {json_file}")

    embeddings = []

    chunks = content["chunks"]
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]

        texts = [c["text"] for c in batch] #comprehension of list of texts from batch of chunks

        batch_embeddings = create_embedding(texts)

        if len(batch_embeddings) != len(batch):
            print(f"Batch failed in {json_file}")
            continue

        embeddings.extend(batch_embeddings)

    if len(embeddings) != len(chunks):
        print(f"Embedding generation incomplete for {json_file}")
        continue

    for chunk, embedding in zip(chunks, embeddings):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embedding

        my_dicts.append(chunk)
        chunk_id += 1

df = pd.DataFrame.from_records(my_dicts)
print(df.head())
joblib.dump(df, 'embeddings.joblib')