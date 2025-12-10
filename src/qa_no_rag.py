from typing import Iterator
from llama_cpp import Llama
import os
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

stream = model.create_chat_completion(
    messages=[
        {
        'role':'system',
        'content': 'You are a helpful assistant who replies truthfully to user questions.'
        },
        {
            'role': 'user',
            'content': input(),
        }
    ],
    max_tokens=500,
    temperature=0.8,
    stream=True,
)

# L'output è un iteratore
assert isinstance(stream, Iterator)
# Stampa l'iteratore a schermo un token alla volta, e ritorna la sequenza finale
assistant_message = print_stream(stream)
