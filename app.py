import json
import os
import time
from typing import Literal
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import requests
import streamlit as st


# 1. Defined Schema
class TicketClassification(BaseModel):
  category: Literal[
      "Billing", "Technical", "Account Access", "General Inquiry"
  ]
  urgency: Literal["Low", "Medium", "High", "Critical"]
  action_required: str = Field(
      description="Recommended next step for support agent"
  )
  summary: str = Field(description="One sentence issue summary")


st.set_page_config(page_title="AI Support Ticket Classifier", page_icon="🤖")

st.title("🤖 AI Support Ticket Classifier")
st.write(
    "Enter a customer support request below to automatically classify its"
    " department and priority."
)

ticket_input = st.text_area(
    "Customer Request:",
    placeholder="e.g., I was charged twice for my subscription this month!",
)

# Read API Keys from Environment Variables
gemini_api_key = os.environ.get("GEMINI_API_KEY")
hf_token = os.environ.get("HF_TOKEN")


# --- Gemini Engine with Auto-Retry on 503 ---
def classify_with_gemini(
    client: genai.Client, text: str, retries: int = 3
) -> str:
  for attempt in range(retries):
    try:
      response = client.models.generate_content(
          model="gemini-3.8-flash",
          contents=f"Classify this support ticket:\n\n{text}",
          config=types.GenerateContentConfig(
              response_mime_type="application/json",
              response_schema=TicketClassification.model_json_schema(),
              temperature=0.1,
          ),
      )
      return response.text
    except Exception as e:
      # If 503 high demand spike, sleep briefly and retry
      if ("503" in str(e) or "UNAVAILABLE" in str(e)) and attempt < retries - 1:
        time.sleep(1.5 * (attempt + 1))
        continue
      raise e


# --- Fallback Engine: Updated Hugging Face Serverless Router ---
def classify_with_huggingface(prompt_text: str, token: str) -> str:
  url = (
      "https://router.huggingface.co/hf-inference/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions"
  )
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
  }

  system_prompt = (
      "You are a strict customer support ticket classifier. Reply ONLY with a"
      ' valid JSON matching: {"category": "Billing"|"Technical"|"Account'
      ' Access"|"General Inquiry", "urgency": "Low"|"Medium"|"High"|"Critical",'
      ' "summary": "<1-sentence summary>", "action_required": "<recommended'
      ' next action>"}. No markdown backticks or explanations.'
  )

  payload = {
      "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt_text},
      ],
      "max_tokens": 250,
      "temperature": 0.1,
  }

  response = requests.post(url, headers=headers, json=payload, timeout=12)

  if response.status_code == 200:
    data = response.json()
    raw_content = data["choices"][0]["message"]["content"].strip()
    return raw_content.replace("```json", "").replace("```", "").strip()
  else:
    raise Exception(
        f"Hugging Face HTTP {response.status_code}: {response.text}"
    )


# --- Execution Pipeline ---
if st.button("Classify Ticket", type="primary"):
  if not ticket_input.strip():
    st.warning("Please enter a ticket message first.")
  else:
    classified_data = None
    used_provider = None

    # Step 1: Try Gemini with Exponential Backoff
    if gemini_api_key:
      try:
        client = genai.Client(api_key=gemini_api_key)
        with st.spinner("Analyzing ticket with Gemini..."):
          raw_json = classify_with_gemini(client, ticket_input)
          classified_data = TicketClassification.model_validate_json(raw_json)
          used_provider = "Google Gemini API (gemini-3.8-flash)"
      except Exception as gemini_err:
        st.warning(
            f"Gemini API issue ({gemini_err}). Switching to Hugging Face"
            " Fallback..."
        )

    # Step 2: Fallback to Hugging Face
    if not classified_data and hf_token:
      try:
        with st.spinner("Analyzing ticket with Hugging Face Inference..."):
          hf_raw_json = classify_with_huggingface(ticket_input, hf_token)
          classified_data = TicketClassification.model_validate_json(
              hf_raw_json
          )
          used_provider = "Hugging Face Serverless Router"
      except Exception as hf_err:
        st.error(f"Hugging Face Fallback Error: {hf_err}")

    # Step 3: Render Output
    if classified_data:
      st.divider()
      col1, col2 = st.columns(2)
      col1.metric("Department", classified_data.category)
      col2.metric("Urgency", classified_data.urgency)

      st.subheader("Summary")
      st.info(classified_data.summary)

      st.subheader("Recommended Action")
      st.success(classified_data.action_required)

      st.caption(f"⚡ Processed using: **{used_provider}**")
    elif not gemini_api_key and not hf_token:
      st.error("Please configure API keys in Render Environment Variables.")
