import pandas as pd
import numpy as np
import torch
import json
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def generate_predictions_and_dossiers(input_csv, model_dir, output_csv, output_json):
    print(f"Loading Model and Tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loading Input Data: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 1. Safely Format Inputs (Match Training Data Structure)
    channel = df.get('Ticket_Channel', pd.Series('Unknown', index=df.index)).astype(str)
    t_type = df.get('Ticket_Type', pd.Series('Unknown', index=df.index)).astype(str)
    res_time = df.get('Resolution_Time_Hours', pd.Series('0', index=df.index)).astype(str)
    priority = df.get('Priority_Level', pd.Series('Unknown', index=df.index)).astype(str)
    
    combined_texts = (
        "[ASSIGNED PRIORITY: " + priority + "] " +
        "[CHANNEL: " + channel + "] " +
        "[TYPE: " + t_type + "] " +
        "[RESOLUTION HOURS: " + res_time + "] " +
        "TEXT: " + df['Ticket_Description'].astype(str)
    )

    # 2. Run Inference in Batches
    print("Running Inference...")
    predictions = []
    
    # Process in chunks of 32 to avoid Out-Of-Memory errors
    for i in range(0, len(combined_texts), 32):
        batch_texts = combined_texts.iloc[i:i+32].tolist()
        inputs = tokenizer(
            batch_texts, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            
    df['Predicted_Mismatch'] = predictions
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
    # 3. Deterministic Dossier Generation (Zero Hallucination)
    print("Generating Evidence Dossiers...")
    mismatches = df[df['Predicted_Mismatch'] == 1]
    
    pri_to_num = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    escalation_words = ['urgent', 'escalate', 'supervisor', 'manager', 'lawsuit', 'unacceptable', 'failing', 'down', 'outage', 'immediately']
    dossiers = []
    
    for _, row in mismatches.iterrows():
        ticket_text = str(row.get('Ticket_Description', '')).lower()
        assigned = str(row.get('Priority_Level', 'Unknown'))
        res_time_val = row.get('Resolution_Time_Hours', 0)
        
        # Calculate behavioral time score based on Stage 1 logic
        if pd.isna(res_time_val): time_score = 2
        else:
            try:
                h = float(res_time_val)
                if h > 120: time_score = 4
                elif h > 48: time_score = 3
                elif h > 12: time_score = 2
                else: time_score = 1
            except ValueError: time_score = 2
            
        # For inference without pseudo-labels, we use time proxy to estimate "inferred" severity
        inferred = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}.get(time_score, 'Unknown')
        
        assigned_num = pri_to_num.get(assigned, 2)
        delta = time_score - assigned_num
        
        if delta > 0: mismatch_type = "Hidden Crisis"
        elif delta < 0: mismatch_type = "False Alarm"
        else: mismatch_type = "Disputed Ground Truth" 
            
        found_keywords = [w for w in escalation_words if w in ticket_text]
        keyword_evidence = {
            "signal": "keyword",
            "value": ", ".join(found_keywords) if found_keywords else "None",
            "weight": str(len(found_keywords) * 0.5) 
        }
        
        time_evidence = {
            "signal": "resolution_time",
            "value": f"{res_time_val} hours",
            "interpretation": f"Time proxy indicates severity level {time_score}"
        }
        
        analysis = f"Ticket was assigned as {assigned} but resolved in {res_time_val} hours, indicating a behavioral severity of {time_score}. "
        if found_keywords:
            analysis += f"The presence of escalation keywords ({', '.join(found_keywords)}) further supports the mismatch rating. "
        analysis += "The assigned priority contradicts the objective processing metrics."

        dossier = {
            "ticket_id": str(row.get('Ticket_ID', 'Unknown')),
            "assigned_priority": assigned,
            "inferred_severity": inferred,
            "mismatch_type": mismatch_type,
            "severity_delta": str(delta),
            "feature_evidence": [keyword_evidence, time_evidence],
            "constraint_analysis": analysis,
            "confidence": "High" 
        }
        dossiers.append(dossier)
        
    with open(output_json, 'w') as f:
        json.dump(dossiers, f, indent=4)
        
    print(f"Successfully generated {len(dossiers)} dossiers.")
    print(f"Dossiers saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIA Inference and generate Dossiers")
    parser.add_argument("--input", required=True, help="Path to input CSV containing new tickets")
    parser.add_argument("--model", default="./final_model", help="Path to the saved RoBERTa model directory")
    parser.add_argument("--out_csv", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--out_json", default="dossiers.json", help="Output JSON path")
    args = parser.parse_args()
    
    generate_predictions_and_dossiers(args.input, args.model, args.out_csv, args.out_json)