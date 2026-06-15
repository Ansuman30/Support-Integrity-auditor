# Support Integrity Auditor (SIA) - MARS Open Projects 2026

## Project Overview
The Support Integrity Auditor (SIA) is an automated triage evaluation system designed to detect "Priority Mismatches" in customer support tickets. It identifies cases where the objective characteristics of a ticket (text, resolution time, channel) conflict with the human-assigned priority level.

## Methodology & Architecture
The system operates in three distinct stages:
1. **Self-Supervised Pseudo-Labeling:** A zero-shot LLM pipeline combined with a resolution-time proxy to infer the true severity of 20,000 tickets, generating binary mismatch labels.
2. **Classifier Fine-Tuning:** A RoBERTa-base model fine-tuned on the pseudo-labeled dataset. Text features were fused with structured metadata (Channel, Type, Resolution Time). Class imbalance was mitigated using square-root dampened weighted CrossEntropy loss.
3. **Deterministic Dossier Generation:** A rule-based NLP extraction pipeline that generates zero-hallucination evidence dossiers explaining the reasoning behind every detected mismatch.

## Stage 1 Ablation Study
The following table justifies the signal fusion strategy used to generate pseudo-labels:

| Signal | Match Rate (vs Original Priority) |
| :--- | :--- |
| LLM-Only (Phi-3) | 30.35% |
| Resolution Time | 26.25% |
| **Fused Pipeline (0.7 / 0.3)** | **29.45%** |
*Note: Low match rates indicate high noise/inconsistency in the original human labels, confirming the necessity of the auditor.*

## Final Verification Metrics (Stage 3)
| Metric | Target | Final Score |
| :--- | :--- | :--- |
| Binary Classification Accuracy | >= 83% | **94.3** |
| Macro F1 Score | >= 0.82 | **0.933** |
| Per-Class Recall (Consistent) | >= 0.78 | **0.9542** |
| Per-Class Recall (Mismatch) | >= 0.78 | **0.9383** |

## Setup Instructions
1. Clone the repository: `git clone https://github.com/Ansuman30/Support-Integrity-auditor.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the dashboard: `streamlit run app.py`

Additionally added streamlit url to view
