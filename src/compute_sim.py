from llama_cpp import Llama
import os
import numpy as np
import sys
from sim import sim
import silence


model = Llama(
    # Path al file GGUF del modello
    model_path=os.path.expanduser('~/models/granite-embedding-30m-english-f16.gguf'),
    # Silenzia i logs di questo modello
    verbose=False,
    # Modello embedding
    embedding=True,
    # Massima dimensione della context window (in numero di token)
    # n_batch=512,
    # n_ctx=1024,
    # Per riproducibilità
    seed=0,
)

def embed(model: Llama, texts: list[str]):

    results = model.create_embedding(texts)
    embeddings = [x['embedding'] for x in results['data']]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings

if __name__ == '__main__':

    embeddings = embed(model, [sys.argv[1], sys.argv[2]])
    print(sim(embeddings[0], embeddings[1]))

