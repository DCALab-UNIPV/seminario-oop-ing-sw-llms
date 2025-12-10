from llama_cpp import Llama
import os
import silence


model = Llama(
    # Path al file GGUF del modello
    model_path=os.path.expanduser('~/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf'),
    # Silenzia i logs di questo modello
    verbose=False,
    # Dimensione della context window data in numero di token
    n_ctx=512,
    # Per riproducibilità
    seed=0,
)

stringa = 'The quick brown fox jumped over the lazy dog, discombulation.'

# Tokenizzazione
buffer = stringa.encode('utf-8')
token_ids = model.tokenize(buffer)

# Detokenizzazione
for tokid in token_ids:
    print(model.detokenize([tokid]))

