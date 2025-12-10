from typing import Iterator, List
from llama_cpp import ChatCompletionRequestMessage, Llama
import os
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from print_stream import print_stream
import silence


model = Llama(
    # Path al file GGUF del modello
    model_path='models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    # Formato chat: attenzione a selezionare quello giusto, altrimenti output non ha senso
    chat_format='qwen',
    # Silenzia i logs di questo modello
    verbose=False,
    # Massima dimensione della context window (in numero di token)
    n_ctx=1024,
    # Per riproducibilità
    seed=0,
)

# Inizializza la conversazione con un messaggio "di sistema"
messages: List[ChatCompletionRequestMessage] = [
    {
        'role': 'system',
        'content': 'You are a helpful and witty assistant who knows how to crack jokes.'
    }
]

# Per salvare la cronologia della shell interattiva
session = PromptSession(
    history = FileHistory(os.path.expanduser('~/.oop-chatbot-history')),
)

while True:

    # Attendi l'input dell'utente
    user_input = session.prompt('> ').strip()

    # Evita di invocare il modello inutilmente
    if not user_input:
        continue

    # Aggiungi un nuovo messaggio da parte dell'utente in coda alla chat
    messages.append({
        'role': 'user',
        'content': user_input,
    })

    # Ri-passa TUTTA la conversazione al modello!
    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=500,
        temperature=0.8,
        stream=True,
    )

    # L'output è un iteratore
    assert isinstance(stream, Iterator)
    # Stampa l'iteratore a schermo un token alla volta, e ritorna la sequenza finale
    assistant_message = print_stream(stream)
    
    # Aggiungi il messaggio dell'assistente completo in coda alla chat
    messages.append({
        'role': 'assistant',
        'content': assistant_message, 
    })
    
    # A capo
    print()

