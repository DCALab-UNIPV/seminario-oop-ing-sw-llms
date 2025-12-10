#!/bin/python
from typing import Iterator
from llama_cpp import Llama
from print_stream import print_stream
import os
import silence


model = Llama(
    # Path al file GGUF del modello
    model_path='models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    # Silenzia i logs di questo modello
    verbose=False,
    # Massima dimensione della context window (in numero di token)
    n_ctx=1024,
    # Per riproducibilità
    seed=0,
    # Formato chat: attenzione a selezionare quello giusto, altrimenti output non ha senso
    chat_format='qwen',
)

stream = model.create_chat_completion(
    messages=[
        {
            'role': 'user',
            'content': (
                'How many fingers do 10 people have? Think step by step.'
            )
        }
    ],
    max_tokens=500,
    temperature=0.1,
    stream=True,
)

assert isinstance(stream, Iterator)
print_stream(stream)
print()
