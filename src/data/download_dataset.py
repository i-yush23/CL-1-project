import pandas as pd
from datasets import load_dataset
import requests
import json
import os

def download_hinglish_data():
    all_data = []

    print("--- Loading L3Cube-HingLID (Social Media/Twitter) via GitHub ---")
    splits = ['train', 'test', 'validation']
    base_url = "https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/main/L3Cube-HingLID/{}.txt"
    
    for split in splits:
        url = base_url.format(split)
        try:
            print(f"Fetching {split} data from {url}...")
            response = requests.get(url)
            response.raise_for_status()
            
            # The file is a tab-separated CoNLL-like format
            lines = response.text.strip().split('\n')
            tokens = []
            labels = []
            
            for line in lines:
                if line.strip() == '':
                    if tokens:
                        all_data.append({
                            "tokens": tokens,
                            "labels": labels,
                            "source": f"l3cube_{split}"
                        })
                        tokens = []
                        labels = []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        # Normalize label to keep consistency (e.g. HI, EN, O, MIXED)
                        labels.append(parts[1].strip())
            
            # Catch last sentence if file doesn't end with empty line
            if tokens:
                all_data.append({
                    "tokens": tokens,
                    "labels": labels,
                    "source": f"l3cube_{split}"
                })
            print(f"Successfully processed {split} split. Current total: {len(all_data)}")
        except Exception as e:
            print(f"L3Cube {split} fetch error: {e}")

    # For sequence tagging (CRF), we require token-level annotations.
    output_file = "hinglish_crf_train_data.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"\nSuccess! Total annotated LID sentences collected for CRF: {len(all_data)}")
    print(f"Data saved to: {output_file}")

    print("\n--- Downloading POS Dataset from Github Mirror (Mysterious19 POS-tagging-Codemix) ---")
    pos_sources = [
        "https://raw.githubusercontent.com/Mysterious19/POS-tagging-Codemix/master/Hi-En_data/FacebookData.txt",
        "https://raw.githubusercontent.com/Mysterious19/POS-tagging-Codemix/master/Hi-En_data/TwitterData.txt",
        "https://raw.githubusercontent.com/Mysterious19/POS-tagging-Codemix/master/Hi-En_data/WhatsApp.txt"
    ]
    
    pos_tokens, pos_labels = [], []
    for url in pos_sources:
        try:
            print(f"Fetching POS data from {url.split('/')[-1]}...")
            response = requests.get(url)
            response.raise_for_status()
            
            lines = response.text.replace('\\r', '').split('\n')
            for line in lines:
                parts = line.strip().split()
                # Format is often: [Token] [LangTag] [POS_Tag]
                if len(parts) >= 3:
                    pos_tokens.append(parts[0])
                    pos_labels.append(parts[2])
                elif line.strip() == '':
                    pos_tokens.append('') # Sentinel for sentence boundary expected by POS CSV parser
                    pos_labels.append('')
            print(f"Successfully processed {url.split('/')[-1]}.")
        except Exception as e:
            print(f"POS data fetch error: {e}")

    if pos_tokens:
        out_dir = os.path.join("data", "raw")
        os.makedirs(out_dir, exist_ok=True)
        pos_csv_file = os.path.join(out_dir, "train_pos.csv")
        df = pd.DataFrame({"token": pos_tokens, "label": pos_labels})
        df.to_csv(pos_csv_file, index=False)
        print(f"\nSuccess! POS data collected and saved to: {pos_csv_file}")
    
if __name__ == "__main__":
    download_hinglish_data()