#!/bin/python
from typing import Iterator
from llama_cpp import Llama
from print_stream import print_stream
import os
import silence


model = Llama(
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

stream = model.create_chat_completion(
    messages=[
        {
            'role': 'system',
            'content': (
                'You are a professional summarizer.\n' +
                'You will summarize the input text IN ONE SINGLE SENTENCE.\n' + 
                'Make it VERY SHORT. One sentence is ENOUGH!' 
            )
        },
        {
            'role': 'user',
            'content': 'Summarize the following text in one sentence: ' + input()
        }
    ],
    max_tokens=500,
    temperature=0.2,
    stream=True,
)

assert isinstance(stream, Iterator)
print_stream(stream)
print()
