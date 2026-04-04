import pandas as pd
import ollama
import json
import os
import re
import sys
import io

# Force UTF-8 for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CONFIG ---
DB_SIZE = 100  
QA_CSV = 'medquad_qa.csv'
DESC_CSV = 'symptom_Description.csv'
PREC_CSV = 'symptom_precaution.csv'
DATASET_CSV = 'dataset.csv'

def extract_disease_info(disease_name, context_text):
    prompt = f"""
    Based on the medical text provided for '{disease_name}', extract the following in JSON format:
    {{
        "description": "Short 1-2 sentence description",
        "symptoms": ["list", "of", "4-7", "key", "symptoms"],
        "precautions": ["precaution 1", "precaution 2", "precaution 3", "precaution 4"]
    }}
    
    TEXT:
    {context_text}
    
    Only return raw JSON. No extra text.
    """
    try:
        response = ollama.generate(model='phi3', prompt=prompt)
        match = re.search(r'\{.*\}', response['response'], re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"Error for {disease_name}: {e}")
    return None

def main():
    print("Starting Database Expansion (Incremental)...")
    if not os.path.exists(QA_CSV):
        print(f"Error: {QA_CSV} not found.")
        return

    df_qa = pd.read_csv(QA_CSV)
    pattern = re.compile(r'What (?:is|are|is \(are\)) (?:the symptoms of )?(.+?)\?', re.IGNORECASE)
    
    disease_dict = {}
    for _, row in df_qa.iterrows():
        match = pattern.search(str(row['Question']))
        if match:
            d_name = match.group(1).strip()
            if d_name not in disease_dict:
                disease_dict[d_name] = ""
            disease_dict[d_name] += str(row['Answer']) + "\n"
            if len(disease_dict) >= DB_SIZE:
                break
    
    final_diseases = list(disease_dict.keys())
    print(f"Found {len(final_diseases)} diseases to process.")
    
    # Load columns for dataset.csv to ensure consistency
    dataset_cols = pd.read_csv(DATASET_CSV, nrows=0).columns.tolist()

    for i, d_name in enumerate(final_diseases):
        print(f"[{i+1}/{len(final_diseases)}] Processing: {d_name}...")
        info = extract_disease_info(d_name, disease_dict[d_name][:2500])
        
        if info:
            try:
                # 1. Append Description
                desc_val = info.get('description', '')
                pd.DataFrame([{'Disease': d_name, 'Description': desc_val}]).to_csv(DESC_CSV, mode='a', header=False, index=False)
                
                # 2. Append Precautions
                precs = info.get('precautions', [])
                p_row = [d_name] + (precs[:4] + [""] * (4 - len(precs)))
                pd.DataFrame([p_row]).to_csv(PREC_CSV, mode='a', header=False, index=False)
                
                # 3. Append Dataset (Symptoms)
                syms = info.get('symptoms', [])
                s_row = [d_name] + (syms[:len(dataset_cols)-1] + [None] * (len(dataset_cols)-1 - len(syms)))
                pd.DataFrame([s_row], columns=dataset_cols).to_csv(DATASET_CSV, mode='a', header=False, index=False)
                
                print(f"   Saved: {d_name}")
            except Exception as e:
                print(f"   Failed to save {d_name}: {e}")
        else:
            print(f"   Failed to extract info for {d_name}")
        
    print("Expansion Task Finished.")

if __name__ == "__main__":
    main()
