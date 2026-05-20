import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from load_document import load_all_documents
from nltktext_splitter import split_documents_by_nltk

def initialize_embedding_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

if __name__ == "__main__":
    DB_PATH = "data/vectorstore/faiss_index"
    
    print("Carregando e fatiando documentos locais...")
    raw_docs = load_all_documents("data/raw")
    
    if not raw_docs:
        print("Nenhum documento disponível em 'data/raw' para indexar.")
    else:
        chunks = split_documents_by_nltk(raw_docs)
        print(f" {len(chunks)} chunks gramaticais prontos para indexação.")
        
        embedding_engine = initialize_embedding_model()
        
        print(f"\nGerando embeddings e construindo o índice FAISS...")
        start_time = time.time()
        
        vector_store = FAISS.from_documents(chunks, embedding=embedding_engine)
        
        end_time = time.time()
        print(f"Índice FAISS construído com sucesso em {end_time - start_time:.4f} segundos.")
        
        print(f"Salvando o índice localmente em: {DB_PATH}")
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        vector_store.save_local(DB_PATH)

        query = "critérios de dimensionamento de pontes e estruturas de concreto" 
        
        print("\n" + "="*50)
        print(f"Executando teste de busca semântica no FAISS")
        print(f"Query de teste: '{query}'")
        print("="*50 + "\n")
        
        search_start = time.time()

        results = vector_store.similarity_search(query, k=3)
        search_end = time.time()
        
        print(f"Busca concluída em {(search_end - search_start) * 1000:.2f} ms.\n")
        

        for i, doc in enumerate(results):
            source_file = doc.metadata.get("source", "Desconhecida")
            print(f"Resultado {i+1} (Origem: {source_file}):")
            print(f"Conteúdo Recuperado:\n{doc.page_content.strip()}")
            print("-" * 50 + "\n")