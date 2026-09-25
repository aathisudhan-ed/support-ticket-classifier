import os
import streamlit as st
from google import genai
from pydantic import BaseModel, Field
from typing import Literal

# Defined Schema
class TicketClassification(BaseModel):
    category: Literal["Billing", "Technical", "Account Access", "General Inquiry"]
    urgency: Literal["Low", "Medium", "High", "Critical"]
    action_required: str = Field(description="Recommended next step for support agent")
    summary: str = Field(description="One sentence issue summary")

st.title("🤖 AI Support Ticket Classifier")
ticket_input = st.text_area("Customer Request:")

api_key = os.environ.get("GEMINI_API_KEY")

if st.button("Classify Ticket"):
    if not api_key:
        st.error("Missing GEMINI_API_KEY in Environment Variables!")
    else:
        try:
            client = genai.Client(api_key=api_key)
            
            # Using client.interactions.create
            interaction = client.interactions.create(
                model="gemini-3.8-flash",
                input=f"Classify this support ticket into structured JSON:\n\n{ticket_input}",
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": TicketClassification
                }
            )
            
            # Parse response text
            result = TicketClassification.model_validate_json(interaction.output_text)
            
            st.success(f"Category: {result.category} | Urgency: {result.urgency}")
            st.info(f"Summary: {result.summary}")
            
        except Exception as e:
            st.error(f"Error: {e}")
