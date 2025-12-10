from typing import Iterator
from llama_cpp import CreateChatCompletionStreamResponse
import sys
import silence


def print_stream(stream: Iterator[CreateChatCompletionStreamResponse]):

    # Inizializza il messaggio dell'assistente
    assistant_message = ''
    
    # Per ciascuno step di generazione...
    for item in stream:
        delta = item['choices'][0]['delta']
        
        if 'content' not in delta:
            continue
        
        # Ottieni il token generato a questo step
        token = delta['content']

        if not token:
            continue

        # Stampa il token senza separatori ed immediatamente
        print(token, sep='', end='')
        sys.stdout.flush()
        # Aggiungi il token al messaggio dell'assistente in fieri
        assistant_message += token
    
    return assistant_message
    

    