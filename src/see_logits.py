from llama_cpp import Llama, LogitsProcessorList
import os
from softmax import softmax
import silence


model = Llama(
    # Path al file GGUF del modello
    model_path='models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    # Silenzia i logs di questo modello
    verbose=False,
    # Dimensione della context window data in numero di token
    n_ctx=512,
    # Per riproducibilità
    seed=0,
)

# Lista degli ID dei token di questo modello
all_token_ids = list(range(model.n_vocab()))

# Vedi utilizzo di top k sotto
top_k = 10

def inspect_logits(sequence_token_ids, scores):

    # Converti punteggi/logits dei token in probabilità
    probs = softmax(scores)
    # Tieni solo i token candidati
    entries = sorted(zip(all_token_ids, probs), key=lambda e: e[1], reverse=True)
    entries = entries[:top_k]
    # Converti ID dei token in testo
    sequence = model.detokenize(sequence_token_ids)

    print(sequence.decode('utf-8').replace('\n', '\\n'))
    for i, (tokid, prob) in enumerate(entries, start=1):
        print(i, model.detokenize([tokid]).decode('utf-8').replace('\n', '\\n'), prob)
    print('-' * 8)

    return scores

output = model.create_completion(
    'What is the capital of France?',
    # Massimo numero di token da generare prima di fermarsi
    max_tokens=10,
    # Per sondare o influenzare il campionamento dei token
    logits_processor=LogitsProcessorList([inspect_logits]),
    # Scegli candidati dai primi K token con probabilità più alta
    top_k=top_k,
    # Scegli candidati dal più piccolo insieme di token tali che la somma di probabilità >= p 
    # top_p=0.1,
    # Alzare la temperatura stimola la selezione di candidati meno probabili
    temperature=0,
)

