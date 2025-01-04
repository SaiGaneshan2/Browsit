from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = torch.tensor([])

if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_response(user_input):
    chat_history_ids = st.session_state.chat_history_ids
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.shape[0] > 0 else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    st.session_state.chat_history_ids = chat_history_ids
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

st.title("My name is Browsit_bot!I like talking to you !")
st.write("Lets start talking")

# Input field for user to type their message
user_input = st.text_input("You: ")

if user_input:
    # Store user message in chat history
    st.session_state.messages.append(f"You: {user_input}")

    # Get response from the model
    response = get_response(user_input)

    # Store bot response in chat history
    st.session_state.messages.append(f"Browsit_Bot: {response}")

# Display chat history continuously
for message in st.session_state.messages:
    st.write(message)

