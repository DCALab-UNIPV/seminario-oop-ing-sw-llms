import os
import streamlit as st
from typing import Iterator
from llama_cpp import Llama
import os
import silence


def get_token_stream(stream: Iterator):

    for resp in stream:

        delta = resp['choices'][0]['delta']

        if 'content' in delta:
            content = delta['content']
            if content is None:
                continue
            yield content
        else:
            yield ''


# Initializza conversazione
if 'messages' not in st.session_state:
    
    st.session_state['messages'] = [
        {
            'role': 'system',
            'content': 'You are a helpful and witty assistant who knows how to crack jokes.'
        }
    ]

# Initializza modello
if 'model' not in st.session_state:

    st.session_state['model'] = Llama(
        # Path al file GGUF del modello
        model_path=os.path.expanduser('~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'),
        # Silenzia i logs di questo modello
        verbose=False,
        # Massima dimensione della context window (in numero di token)
        n_ctx=1024,
        # Per riproducibilità
        seed=0,
        # Formato chat: attenzione a selezionare quello giusto, altrimenti output non ha senso
        chat_format='qwen',
    )

# Mostra la cronologia dei messaggi
for message in st.session_state['messages']:

    # Evita di mostrare il messaggio di sistema
    if message['role'] == 'system':
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# When the user sends an input
if user_input := st.chat_input("Ask a question"):

    # Immediately draw the user input on screen (TODO isn't there a way to simply add it have it automatically redrawn?)
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state['messages'].append({
        'role': 'user',
        'content': user_input,
    })
    
    # Draw the partial assistant response
    with st.chat_message("assistant"):

        stream = st.session_state['model'].create_chat_completion(
            messages=st.session_state['messages'],
            max_tokens=500,
            temperature=0.8,
            stream=True,
        )

        response_text = st.write_stream(get_token_stream(stream))
    
        st.session_state['messages'].append({
            'role': 'assistant',
            'content': response_text,
        })
