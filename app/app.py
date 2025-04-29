import streamlit as st
from chatbot import get_answer

st.set_page_config(page_title="Quiz Chatbot 🧠", page_icon="🧠", layout="centered")

# App title
st.title("🧠 Quiz Chatbot")
st.subheader("Ask me anything related to quizzes, and I'll try my best!")

# Initialize session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


with st.container():
    question = st.text_input("Type your quiz question here 👇", placeholder="E.g., What is the capital of France?")

    col1, col2 = st.columns([1, 1])
    with col1:
        submit = st.button("Get Answer 🚀")
    with col2:
        clear = st.button("Clear Chat 🗑️")

# Handle clear button
if clear:
    st.session_state.chat_history = []

# Handle submit button
if submit and question:
    with st.spinner("Thinking... 🤔"):
        answer = get_answer(question)
    st.session_state.chat_history.append((question, answer))

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("🗨️ Chat History")
    for q, a in reversed(st.session_state.chat_history):  # Show latest first
        with st.chat_message("user"):
            st.markdown(f"**Q:** {q}")
        with st.chat_message("assistant"):
            st.markdown(f"**A:** {a}")
