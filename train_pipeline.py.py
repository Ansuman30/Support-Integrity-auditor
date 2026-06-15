import os
import argparse
import pandas as pd
import numpy as np
import torch
import gc
from tqdm.auto import tqdm
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    logging as hf_logging
)

# Suppress warnings for cleaner console output
hf_logging.set_verbosity_error()

# =========================================================
# STAGE 1: PSEUDO-LABEL GENERATION (Phi-3 Auditor)
# =========================================================
def run_stage_1_pseudo_labeling(input_csv):
    print("\n--- STAGE 1: PSEUDO-LABEL GENERATION ---")
    df = pd.read_csv(input_csv).copy()
    
    print("Loading Native Phi-3-mini...")
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token 

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )

    print("Preparing messages for GPU Batching...")
    all_prompts = []
    for text in df['Ticket_Description'].fillna(""):
        messages = [
            {"role": "system", "content": "You are a strict data auditor. Read the support ticket and reply with EXACTLY ONE NUMBER indicating severity: 1 (Low), 2 (Medium), 3 (High), or 4 (Critical). Do not provide any other text."},
            {"role": "user", "content": f"Ticket: '{text}'"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)

    print("Running High-Speed LLM Inference via PyTorch...")
    all_llm_scores = []
    batch_size = 16

    for i in tqdm(range(0, len(all_prompts), batch_size), desc="Scoring Tickets"):
        batch_prompts = all_prompts[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=5, do_sample=False, 
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, use_cache=True
            )
        
        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, prompt_length:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for response in responses:
            response = response.strip()
            score = 2 # Default fallback
            for num in ['4', '3', '2', '1']:
                if num in response: 
                    score = int(num)
                    break
            all_llm_scores.append(score)

    df['score_llm'] = all_llm_scores

    # Clear Phi-3 from GPU memory
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print("Fusing signals (LLM + Resolution Time)...")
    def signal_resolution_time(hours):
        if pd.isna(hours): return 2 
        try: h = float(hours)
        except ValueError: return 2
        if h > 120: return 4    
        elif h > 48: return 3   
        elif h > 12: return 2   
        else: return 1  

    df['score_time'] = df['Resolution_Time_Hours'].apply(signal_resolution_time)
    df['fused_score'] = (df['score_llm'] * 0.7) + (df['score_time'] * 0.3)
    df['inferred_severity_num'] = df['fused_score'].round().astype(int)

    severity_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}
    df['inferred_severity'] = df['inferred_severity_num'].map(severity_map)
    df['Mismatch'] = np.where(df['Priority_Level'] != df['inferred_severity'], 1, 0)
    
    return df

# =========================================================
# CUSTOM TRAINER FOR CLASS IMBALANCE
# =========================================================
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        active_weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=active_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1).long())
        return (loss, outputs) if return_outputs else loss

# =========================================================
# STAGE 2: CLASSIFIER TRAINING (RoBERTa)
# =========================================================
def run_stage_2_training(df, output_model_dir):
    print("\n--- STAGE 2: CLASSIFIER TRAINING ---")
    
    # 1. Metadata Fusion
    channel = df.get('Ticket_Channel', pd.Series('Unknown', index=df.index)).astype(str)
    t_type = df.get('Ticket_Type', pd.Series('Unknown', index=df.index)).astype(str)
    res_time = df.get('Resolution_Time_Hours', pd.Series('0', index=df.index)).astype(str)
    priority = df.get('Priority_Level', pd.Series('Unknown', index=df.index)).astype(str)

    df['combined_text'] = (
        "[ASSIGNED PRIORITY: " + priority + "] " +
        "[CHANNEL: " + channel + "] " +
        "[TYPE: " + t_type + "] " +
        "[RESOLUTION HOURS: " + res_time + "] " +
        "TEXT: " + df['Ticket_Description'].astype(str)
    )
    df['label'] = df['Mismatch']

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # 2. Dampened Class Weights
    base_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    dampened_weights = np.sqrt(base_weights)
    weights_tensor = torch.tensor(dampened_weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. Tokenization
    print("Loading RoBERTa Tokenizer...")
    model_id = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize_function(examples):
        return tokenizer(examples["combined_text"], padding="max_length", truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    # 4. Initialization & Metrics
    print("Initializing RoBERTa Model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

    # 5. Training Execution
    training_args = TrainingArguments(
        output_dir="./training_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,                     
        weight_decay=0.01,
        fp16=True if torch.cuda.is_available() else False,                             
        warmup_ratio=0.1,                      
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=True 
    )

    trainer = WeightedLossTrainer(
        class_weights=weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,             
        compute_metrics=compute_metrics,
    )

    print("Starting Training...")
    trainer.train()

    # 6. Save Final Artifacts
    print(f"Saving fine-tuned model to {output_model_dir}...")
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("Pipeline Execution Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Reproducible SIA Training Pipeline")
    parser.add_argument("--input_csv", required=True, help="Path to raw customer_support_tickets.csv")
    parser.add_argument("--model_dir", default="./final_model", help="Directory to save the trained RoBERTa model")
    args = parser.parse_args()

    # Run Stage 1
    pseudo_labeled_df = run_stage_1_pseudo_labeling(args.input_csv)
    
    # Run Stage 2
    run_stage_2_training(pseudo_labeled_df, args.model_dir)