import streamlit as st
from pydantic import BaseModel, Field
from typing import Literal
from transformers import pipeline

# 1. Pydantic Schema for Structured Display
class TicketClassification(BaseModel):
    category: Literal["Billing", "Technical", "Account Access", "General Inquiry"]
    urgency: Literal["Low", "Medium", "High", "Critical"]
    action_required: str
    summary: str

st.set_page_config(page_title="AI Support Ticket Classifier (Offline)", page_icon="🤖")

st.title("🤖 AI Support Ticket Classifier (Offline Engine)")
st.write("Runs 100% locally on CPU without external API calls or rate limits.")

# Load local Zero-Shot Classifier cached in Streamlit memory
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

ticket_input = st.text_area("Customer Request:", placeholder="e.g., I was charged twice for my subscription this month!")

if st.button("Classify Ticket", type="primary"):
    if not ticket_input.strip():
        st.warning("Please enter a ticket message first.")
    else:
        with st.spinner("Classifying ticket locally..."):
            classifier = load_classifier()

            # 1. Predict Category
            categories = ["Billing", "Technical", "Account Access", "General Inquiry"]
            cat_res = classifier(ticket_input, candidate_labels=categories)
            top_category = cat_res["labels"][0]

            # 2. Predict Urgency
            urgencies = ["Low", "Medium", "High", "Critical"]
            urg_res = classifier(ticket_input, candidate_labels=urgencies)
            top_urgency = urg_res["labels"][0]

            # 3. Formulate Summary and Action
            summary_text = ticket_input if len(ticket_input) < 100 else ticket_input[:97] + "..."
            
            action_map = {
                "Billing": "Route ticket to Payments & Refunds team for account audit.",
                "Technical": "Assign ticket to Tier-2 Engineering support for troubleshooting.",
                "Account Access": "Send identity verification link and trigger password reset workflow.",
                "General Inquiry": "Provide automated knowledge-base response or route to general support queue."
            }

            result = TicketClassification(
                category=top_category,
                urgency=top_urgency,
                summary=summary_text,
                action_required=action_map.get(top_category, "Assign to available support agent.")
            )

        # UI Display
        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Department", result.category)
        col2.metric("Urgency", result.urgency)

        st.subheader("Summary")
        st.info(result.summary)

        st.subheader("Recommended Action")
        st.success(result.action_required)

        st.caption("⚡ Processed locally using **Offline Zero-Shot BART Pipeline**")
