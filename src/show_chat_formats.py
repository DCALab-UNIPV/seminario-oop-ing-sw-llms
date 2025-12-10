from typing import List
from llama_cpp import ChatCompletionRequestMessage, Llama
import os
import silence


model = Llama(
    # Path al file GGUF del modello
    model_path='models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    # Formato chat: attenzione a selezionare quello giusto, altrimenti output non ha senso
    chat_format='qwen',
    # Silenzia i logs di questo modello
    verbose=False,
    # Dimensione della context window data in numero di token
    n_ctx=512,
    # Per riproducibilità
    seed=0,
)

messages: List[ChatCompletionRequestMessage] = [
    {
        'role': 'user',
        'content': 'What is the capital of France?',
    }
]

output = model.create_chat_completion(
    messages,
    max_tokens=100,
    temperature=0.8,
)

message = output['choices'][0]['message'] # pyright:ignore
print(message)
