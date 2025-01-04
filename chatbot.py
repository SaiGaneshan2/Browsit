'''from transformers import AutoModelForCausalLM, AutoTokenizer
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
'''

#the code above is without usage of translator model


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

model_name = 'Helsinki-NLP/opus-mt-en-hi'
translate_model = MarianMTModel.from_pretrained(model_name)
translate_tokenizer = MarianTokenizer.from_pretrained(model_name)

if 'chat_history_ids' not in st.session_state:
    st.session_state.chat_history_ids = torch.tensor([])

if 'messages' not in st.session_state:
    st.session_state.messages = []

def translate(text):
    tokens = translate_tokenizer(text, return_tensors="pt", padding=True)
    translated = translate_model.generate(**tokens)
    translation = translate_tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation

def get_response(user_input):
    chat_history_ids = st.session_state.chat_history_ids
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids.shape[0] > 0 else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    st.session_state.chat_history_ids = chat_history_ids
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

st.title("My name is Browsit_bot! I like talking to you!")
st.write("Let's start talking!")

user_input = st.text_input("You: ")

if user_input:
    # Store user message in chat history
    st.session_state.messages.append(f"You: {user_input}")

    # Get response from the model
    response = get_response(user_input)

    # Store bot response in chat history
    st.session_state.messages.append(f"Browsit_Bot: {response}")

for message in st.session_state.messages:
    st.write(message)

if st.button('Translate your chat history to hindi'):
    translated_messages = []
    for message in st.session_state.messages:
        if message.startswith("You:"):
            translated_message = translate(message[4:])
            translated_messages.append(f"You (Translated): {translated_message}")
        elif message.startswith("Browsit_Bot:"):
            translated_message = translate(message[14:])
            translated_messages.append(f"Browsit_Bot (Translated): {translated_message}")
        else:
            translated_messages.append(message)

    for translated_message in translated_messages:
        st.write(translated_message)

