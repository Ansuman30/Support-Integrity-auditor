import streamlit as st
import pandas as pd
import plotly.express as px
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

st.set_page_config(page_title="Support Integrity Auditor", layout="wide")

# =========================================================
# 1. LOAD AI MODEL (CACHED FOR SPEED)
# =========================================================
@st.cache_resource
def load_model():
    # Find where app.py lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "final_model")
    
    if not os.path.exists(model_dir):
        return None, None
        
    # Bypass corrupted local tokenizer and pull the clean base version
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    
    # Load custom fine-tuned weights
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

# ---> THIS WAS THE MISSING LINE <---
tokenizer, model = load_model()

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def analyze_ticket(ticket_desc, priority, channel, ticket_type, res_time):
    combined_text = (
        f"[ASSIGNED PRIORITY: {priority}] "
        f"[CHANNEL: {channel}] "
        f"[TYPE: {ticket_type}] "
        f"[RESOLUTION HOURS: {res_time}] "
        f"TEXT: {ticket_desc}"
    )
    
    # Run Inference
    inputs = tokenizer([combined_text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=-1).item()
        
    is_mismatch = (prediction == 1)
    
    # Deterministic Dossier Generation
    pri_to_num = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    escalation_words = ['urgent', 'escalate', 'supervisor', 'manager', 'lawsuit', 'unacceptable', 'failing', 'down', 'outage', 'immediately']
    
    # Time proxy score
    if res_time > 120: time_score = 4
    elif res_time > 48: time_score = 3
    elif res_time > 12: time_score = 2
    else: time_score = 1
        
    inferred = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}.get(time_score, 'Unknown')
    assigned_num = pri_to_num.get(priority, 2)
    delta = time_score - assigned_num
    
    if delta > 0: mismatch_type = "Hidden Crisis"
    elif delta < 0: mismatch_type = "False Alarm"
    else: mismatch_type = "Disputed Ground Truth" 
        
    found_keywords = [w for w in escalation_words if w in ticket_desc.lower()]
    
    analysis = f"Ticket was assigned as {priority} but resolved in {res_time} hours, indicating a behavioral severity of {time_score}. "
    if found_keywords:
        analysis += f"The presence of escalation keywords ({', '.join(found_keywords)}) further supports the mismatch rating. "
    analysis += "The assigned priority contradicts the objective processing metrics."

    dossier = {
        "ticket_id": "DYNAMIC-UI-001",
        "assigned_priority": priority,
        "inferred_severity": inferred,
        "mismatch_type": mismatch_type,
        "severity_delta": str(delta),
        "feature_evidence": [
            { "signal": "keyword", "value": ", ".join(found_keywords) if found_keywords else "None", "weight": str(len(found_keywords) * 0.5) },
            { "signal": "resolution_time", "value": f"{res_time} hours", "interpretation": f"Time proxy indicates severity level {time_score}" }
        ],
        "constraint_analysis": analysis,
        "confidence": "High" 
    }
    
    return is_mismatch, dossier, delta

# =========================================================
# 3. UI LAYOUT
# =========================================================
st.title("🛡️ Support Integrity Auditor (SIA)")
st.markdown("Automated ticket triage evaluation and mismatch detection.")

if tokenizer is None or model is None:
    st.error(f"🚨 Model not found! Please make sure the `final_model` folder is located exactly here: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_model')}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📄 Single Ticket Check", "📂 Batch Upload", "📊 Mismatch Dashboard"])

# --- TAB 1: SINGLE TICKET INPUT ---
with tab1:
    st.header("Analyze a Single Ticket")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ticket Details")
        ticket_desc = st.text_area("Ticket Description", "I am extremely furious and demanding to speak to a manager immediately! I will file a lawsuit if you do not fix the color of my profile picture. This is unacceptable and URGENT!")
        priority = st.selectbox("Assigned Priority", ["Low", "Medium", "High", "Critical"], index=3) # Default Critical for Demo
        channel = st.selectbox("Channel", ["Email", "Web Form", "Chat", "Phone", "Social Media"])
        ticket_type = st.selectbox("Issue Category", ["Billing", "General Inquiry", "Technical Support", "Account Management", "Other"])
        res_time = st.number_input("Resolution Time (Hours)", min_value=0.0, value=2.0)
        
        if st.button("Run Audit", type="primary"):
            with st.spinner("Running RoBERTa Inference..."):
                is_mismatch, dossier, _ = analyze_ticket(ticket_desc, priority, channel, ticket_type, res_time)
            
            if is_mismatch:
                st.error("🚨 Priority Mismatch Detected!")
            else:
                st.success("✅ Ticket Priority is Consistent.")

    with col2:
        st.subheader("Evidence Dossier")
        if 'dossier' in locals():
            st.json(dossier)
        else:
            st.info("Run an audit to generate the evidence dossier.")

# --- TAB 2: BATCH UPLOAD ---
with tab2:
    st.header("Batch CSV Audit")
    uploaded_file = st.file_uploader("Upload Ticket CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head(3))
        
        if st.button("Process Batch"):
            with st.spinner("Running batch inference (this may take a moment)..."):
                # Run inference on all rows
                results = []
                deltas = []
                
                for _, row in df.iterrows():
                    desc = str(row.get('Ticket_Description', ''))
                    pri = str(row.get('Priority_Level', 'Unknown'))
                    chan = str(row.get('Ticket_Channel', 'Unknown'))
                    
                    # Smart mapping: Handles either Issue_Category (Raw CSV) or Ticket_Type
                    ttype = str(row.get('Issue_Category', row.get('Ticket_Type', 'Unknown')))
                    
                    rtime = float(row.get('Resolution_Time_Hours', 24.0)) if not pd.isna(row.get('Resolution_Time_Hours')) else 24.0
                    
                    is_mismatch, _, delta = analyze_ticket(desc, pri, chan, ttype, rtime)
                    results.append(1 if is_mismatch else 0)
                    deltas.append(delta)
                
                df['Predicted_Mismatch'] = results
                df['Severity_Delta'] = deltas
                
                # Save to session state for the dashboard
                st.session_state['batch_df'] = df
                
                st.success(f"Successfully processed {len(df)} tickets!")
                st.download_button(
                    label="Download Predictions CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name="sia_predictions.csv",
                    mime="text/csv",
                )

# --- TAB 3: DASHBOARD ---
with tab3:
    st.header("Priority Mismatch Dashboard")
    
    if 'batch_df' not in st.session_state:
        st.info("👈 Please process a CSV in the 'Batch Upload' tab to populate the dashboard with real data.")
    else:
        dash_df = st.session_state['batch_df']
        mismatches_only = dash_df[dash_df['Predicted_Mismatch'] == 1]
        
        colA, colB = st.columns(2)
        
        # Smart mapping: Find the category column
        cat_col = 'Issue_Category' if 'Issue_Category' in mismatches_only.columns else 'Ticket_Type'
        
        with colA:
            st.subheader("Mismatches by Category")
            if cat_col in mismatches_only.columns:
                cat_counts = mismatches_only.groupby(cat_col).size().reset_index(name='Mismatch_Count')
                fig_bar = px.bar(cat_counts, x=cat_col, y='Mismatch_Count', color=cat_col)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No Category or Ticket_Type column found in uploaded CSV.")
            
        with colB:
            st.subheader("Severity Delta Heatmap")
            if 'Ticket_Channel' in dash_df.columns:
                heatmap_data = pd.crosstab(dash_df['Ticket_Channel'], dash_df['Severity_Delta'])
                fig_heat = px.imshow(heatmap_data, text_auto=True, aspect="auto", 
                                     labels=dict(x="Severity Delta", y="Channel", color="Count"),
                                     color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.warning("Column 'Ticket_Channel' not found in uploaded CSV.")