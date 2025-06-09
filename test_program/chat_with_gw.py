# based onhttps://github.com/rootfs/ai-gateway/blob/auto-model/docs/chat-with-ai-gw.py
import streamlit as st
from openai import OpenAI
from datetime import datetime
import traceback
import time

st.set_page_config(page_title="Semantic Router Chat", layout="wide", page_icon="💬")

# Initialize session states
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "system_prompts" not in st.session_state:
    st.session_state.system_prompts = {}

def create_new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.chats[chat_id] = []
    st.session_state.system_prompts[chat_id] = ""
    st.session_state.current_chat_id = chat_id
    return chat_id

# Sidebar for configuration and chat management
with st.sidebar:
    st.title("Settings")
    api_endpoint = st.text_input("Semantic Router Endpoint", "http://localhost:8801/v1")
    model = st.text_input("Model Name", "auto")
    api_key = st.text_input("API Key (if required)", type="password")
    
    # System prompt input
    st.subheader("System Prompt")
    if st.session_state.current_chat_id:
        system_prompt = st.text_area(
            "Define AI behavior",
            value=st.session_state.system_prompts.get(st.session_state.current_chat_id, ""),
            height=150
        )
        # Update system prompt for current chat
        if st.session_state.current_chat_id:
            st.session_state.system_prompts[st.session_state.current_chat_id] = system_prompt
    
    st.button("New Chat", on_click=create_new_chat)
    
    st.subheader("Chat History")
    for chat_id in reversed(list(st.session_state.chats.keys())):
        timestamp = datetime.strptime(chat_id, "%Y%m%d_%H%M%S")
        if st.button(f"Chat {timestamp.strftime('%Y-%m-%d %H:%M:%S')}", key=chat_id):
            st.session_state.current_chat_id = chat_id

# Ensure there's at least one chat
if not st.session_state.chats:
    create_new_chat()

# Main chat interface
st.title("Chat with Semantic Router")

# Display chat messages for current chat
if st.session_state.current_chat_id:
    for message in st.session_state.chats[st.session_state.current_chat_id]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to discuss?"):
        # Add user message to chat history
        st.session_state.chats[st.session_state.current_chat_id].append(
            {"role": "user", "content": prompt}
        )
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                start_time = time.time()
                # Initialize OpenAI client with custom base URL
                client = OpenAI(base_url=api_endpoint, api_key=api_key if api_key else "dummy")
                
                # Get system prompt for current chat
                system_prompt = st.session_state.system_prompts.get(st.session_state.current_chat_id, "")
                
                # Prepare messages array for the API call
                messages = []
                
                # Add system prompt if present
                if system_prompt:
                    messages.append({"role": "assistant", "content": system_prompt})
                
                messages.append({"role": "user", "content": prompt})
                
                # Debug information
                st.sidebar.write("Debug - Messages being sent:")
                st.sidebar.json(messages)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    stream=False
                )
                end_time = time.time()
                latency = end_time - start_time
                # Extract response content
                print(response)
                full_response = response.choices[0].message.content      
                print(full_response)          
                message_placeholder.write(str(full_response))
                st.write(f"⏱️ Latency: {latency:.2f} seconds")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(traceback.format_exc())
                full_response = "Error occurred while communicating with Semantic Router"
                message_placeholder.write(full_response)
            
            # Add assistant response to chat history
            if full_response:
                st.session_state.chats[st.session_state.current_chat_id].append(
                    {"role": "assistant", "content": full_response}
                )