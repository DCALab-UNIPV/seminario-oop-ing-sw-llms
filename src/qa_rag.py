from typing import Iterator
from llama_cpp import Llama
import os
import numpy as np
from sim import sim
from print_stream import print_stream
import faiss
import silence
import sys


def get_chunks(text: str, size: int, overlap: int):

    '''
    Divide il testo in chunks con dimensione massima (in parole) e sovrapposizione (in parole).
    '''

    # Divide testo in parole
    words = text.split()
    # Inizializza lista dei chunks
    chunks: list[str] = []

    for i in range(0, len(words) - overlap, size - overlap):

        chunk_words = words[i:i + size]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)           
    
    return chunks

def get_chunked_corpus(path: str, size: int, overlap: int):

    'Carica file di testo e lo divide in chunks'

    with open(path) as f:
        return get_chunks(f.read(), size, overlap)

def embed(model: Llama, texts: list[str]):

    results = model.create_embedding(texts)
    embeddings = [x['embedding'] for x in results['data']]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings


class SlowRetriever:

    '''
    Retriever lento che confronta query con ciascun chunk (cosine similarity degli embedding).
    '''

    def __init__(self, path: str, size: int = 50, overlap: int = 5):

        self.encoder = Llama(
            # Path al file GGUF del modello
            model_path='models/granite-embedding-30m-english-f16.gguf',
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
        self.corpus = get_chunked_corpus(path, size, overlap)
        self.embeddings = embed(self.encoder, self.corpus)

    def find(self, query: str, k:int = 5):

        assert len(self.corpus) == len(self.embeddings)
        query_embedding = embed(self.encoder, [query])[0]
        results = [(x, sim(e, query_embedding)) for x, e in zip(self.corpus, self.embeddings)]
        results = sorted(results, key=lambda x: x[1], reverse=True)
        results = results[:k]
        return results


class FAISSRetriever:

    '''
    Retriever veloce che usa FAISS: https://en.wikipedia.org/wiki/FAISS.
    '''

    def __init__(self, path: str, size: int = 50, overlap: int = 5):

        self.encoder = Llama(
            # Path al file GGUF del modello
            model_path='models/granite-embedding-30m-english-f16.gguf',
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
        self.corpus = get_chunked_corpus(path, size, overlap)
        embeddings = embed(self.encoder, self.corpus)
        # Crea indice efficiente FAISS
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)  # pyright:ignore

    def find(self, query: str, k:int = 5):
        
        query_embedding = embed(self.encoder, [query]) if isinstance(query, str) else query
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, len(self.corpus)) # D: similarities, I: indices # pyright:ignore
        results = [(self.corpus[int(idx)], float(sim)) for sim, idx in zip(D[0], I[0])]
        results = results[:k]
        return results


if __name__ == '__main__':

    query = sys.argv[1]
    corpus_path = sys.argv[2]

    decoder = Llama(
        # Path al file GGUF del modello
        model_path='models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
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

    retriever = SlowRetriever(corpus_path, size=50, overlap=5)
    # retriever = FAISSRetriever(corpus_path, size=50, overlap=5)
    chunks_and_scores = retriever.find(query)
    chunks = [c for c , _ in chunks_and_scores]
    context = ' '.join(chunks)

    stream = decoder.create_chat_completion(
        messages=[
            {
            'role':'system',
            'content': (
                'You are a helpful assistant who replies accurately to user questions '+
                'based on the context that the user provides.'
            )
            },
            {
                'role': 'user',
                'content': (
                    'Answer the following question using information from the following context.\n' +
                    'The question:\n' +
                    query + '\n' +
                    'The context:\n' +
                    context + '\n'
                ),
            }
        ],
        max_tokens=500,
        temperature=0.8,
        stream=True,
    )

    assert isinstance(stream, Iterator)
    print_stream(stream)

