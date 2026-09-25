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


# --- Fallback Engine: Hugging Face Router API ---
def classify_with_huggingface(prompt_text: str, token: str) -> str:
  """Calls Hugging Face's serverless router using standard OpenAI-compatible format."""
  url = "https://router.huggingface.co/hf-inference/v1/chat/completions"
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
  }

  system_prompt = (
      "You are a strict customer support ticket classifier. Classify the user"
      " ticket and reply ONLY with a raw valid JSON object matching this schema:"
      ' {"category": "Billing"|"Technical"|"Account Access"|"General'
      ' Inquiry", "urgency": "Low"|"Medium"|"High"|"Critical", "summary":'
      ' "<1-sentence summary>", "action_required": "<recommended next'
      ' action>"}. Do NOT include markdown wrapping or extra text.'
  )

  payload = {
      "model": "meta-llama/Llama-3.2-3B-Instruct",
      "messages": [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt_text},
      ],
      "max_tokens": 300,
      "temperature": 0.1,
  }

  response = requests.post(url, headers=headers, json=payload, timeout=10)

  if response.status_code == 200:
    data = response.json()
    raw_content = data["choices"][0]["message"]["content"].strip()
    return raw_content.replace("```json", "").replace("```", "").strip()
  else:
    raise Exception(
        f"Hugging Face HTTP {response.status_code}: {response.text}"
    )


# --- Execution Logic ---
if st.button("Classify Ticket", type="primary"):
  if not ticket_input.strip():
    st.warning("Please enter a ticket message first.")
  else:
    classified_data = None
    used_provider = None

    # Step 1: Try Gemini
    if gemini_api_key:
      try:
        client = genai.Client(api_key=gemini_api_key)
        with st.spinner("Analyzing ticket with Gemini..."):
          response = client.models.generate_content(
              model="gemini-1.5-flash",  # Reliable stable endpoint
              contents=f"Classify this support ticket:\n\n{ticket_input}",
              config=types.GenerateContentConfig(
                  response_mime_type="application/json",
                  response_schema=TicketClassification.model_json_schema(),
                  temperature=0.1,
              ),
          )
          classified_data = TicketClassification.model_validate_json(
              response.text
          )
          used_provider = "Google Gemini API"
      except Exception as gemini_err:
        st.warning(
            f"Gemini API unavailable ({gemini_err}). Switching to Hugging Face"
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
          used_provider = "Hugging Face Serverless (Llama-3.2)"
      except Exception as hf_err:
        st.error(f"Hugging Face Fallback Error: {hf_err}")

    # Display UI Output
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
      st.error("Please configure your API keys in Render Environment Variables.")
