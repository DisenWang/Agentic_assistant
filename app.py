import asyncio
import streamlit as st
from main import agent 

st.set_page_config(page_title="Chemistry Assistant", layout="centered")

st.markdown(
    "<style>body { background-color: white !important; }</style>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f4f6f8;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessage.user {
        background-color: #e0f7fa;
    }
    .stChatMessage.assistant {
        background-color: #ede7f6;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Chemistry Assistant")

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

user_input = st.chat_input("Ask about a molecule")
if user_input:
    st.session_state.state["input"] = user_input
    st.session_state.state = asyncio.run(agent.ainvoke(st.session_state.state))
    st.session_state.memory = st.session_state.state["memory"]

# Display chat history
with st.container():
    st.subheader("Chat History")
    import os
    from PIL import Image
    import io
    for msg in st.session_state.memory:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        elif msg.type == "ai":
            st.chat_message("assistant").write(msg.content)
            if (
                "Molecule image saved" in msg.content
                and os.path.exists("output.png")
            ):
                with open("output.png", "rb") as f:
                    image_bytes = f.read()
                image = Image.open(io.BytesIO(image_bytes))
                st.image(image, caption="Molecule Visualization", width=300)