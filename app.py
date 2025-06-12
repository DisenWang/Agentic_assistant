import streamlit as st
from main import agent  # assuming temp.py has your LangGraph logic

st.set_page_config(page_title="Chemistry Assistant", layout="centered")
st.title("🧪 Chemistry Assistant")

# Session state for chat
if "memory" not in st.session_state:
    st.session_state.memory = []
if "state" not in st.session_state:
    st.session_state.state = {
        "input": "",
        "memory": [],
        "compound": None,
        "smiles": None,
        "formula": None,
        "descriptors": None
    }

user_input = st.chat_input("Ask about a molecule (e.g., logP of aspirin)")
if user_input:
    st.session_state.state["input"] = user_input
    st.session_state.state = agent.invoke(st.session_state.state)
    st.session_state.memory = st.session_state.state["memory"]

# Display chat history
for msg in st.session_state.memory:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        st.chat_message("assistant").write(msg.content)