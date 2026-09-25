import os
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Literal

# Defined Pydantic Schema
class TicketClassification(BaseModel):
    category: Literal["Billing", "Technical", "Account Access", "General Inquiry"]
    urgency: Literal["Low", "Medium", "High", "Critical"]
    action_required: str = Field(description="Recommended next step for support agent")
    summary: str = Field(description="One sentence issue summary")

st.set_page_config(page_title="AI Support Ticket Classifier", page_icon="🤖")

st.title("🤖 AI Support Ticket Classifier")
st.write("Enter a customer support request below to automatically classify its department and priority.")

# Input Field
ticket_input = st.text_area("Customer Request:", placeholder="e.g., I was charged twice for my subscription this month!")

# API Key handling from Render Environment
api_key = os.environ.get("GEMINI_API_KEY")

if st.button("Classify Ticket", type="primary"):
    if not api_key:
        st.error("Missing GEMINI_API_KEY in Environment Variables!")
    elif not ticket_input.strip():
        st.warning("Please enter a ticket message first.")
    else:
        try:
            client = genai.Client(api_key=api_key)
            with st.spinner("Analyzing ticket with Gemini..."):
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=f"Classify this support ticket:\n\n{ticket_input}",
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TicketClassification.model_json_schema(),
                        temperature=0.1
                    )
                )
                
                # Parse JSON output string back into Pydantic model instance
                result = TicketClassification.model_validate_json(response.text)

            # Display Results
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("Department", result.category)
            col2.metric("Urgency", result.urgency)

            st.subheader("Summary")
            st.info(result.summary)

            st.subheader("Recommended Action")
            st.success(result.action_required)

        except Exception as e:
            st.error(f"Error processing ticket: {e}")
