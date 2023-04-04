import streamlit as st

from chatbot.utils.LoadBert import LoadBert
from chatbot.services.chatservices import *

st.set_page_config(page_title="Yepto Chatbot", page_icon=":robot_face:")
st.title("Welcome to the Yepto Chatbot")
st.markdown("I can help you with your FAQ about university")
st.write("dssd")

model, tokenizer, intent_labels = LoadBert.load_bert(trainable=True)

user_input = st.text_input("Enter your question")
st.write("User:", user_input)
intent_tag, confidence_score = extract_intent(model, tokenizer, intent_labels, user_input)

response = get_response(intent_tag)

if user_input=="":
    st.write("Bot: Hello there! What can I do for you today?")
else:
    st.write("Bot:", response)

