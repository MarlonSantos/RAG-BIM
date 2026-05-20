import time
from langchain_huggingface import HuggingFaceEmbeddings
from load_document import load_all_documents
from nltktext_splitter import split_documents_by_nltk

def initialize_embedding_model():
    print("Carregando modelo de embeddings (HuggingFace)...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        
        model_kwargs={'device': 'cpu'}, # GPU: "cuda"
        encode_kwargs={'normalize_embeddings': True} 
    )
    return model

if __name__ == "__main__":
    print("Processando documentos locais...")
    raw_docs = load_all_documents("data/raw")
    
    if not raw_docs:
        print("Nenhum documento disponível em 'data/raw' para gerar embeddings.")
    else:
        chunks = split_documents_by_nltk(raw_docs)
        print(f" {len(chunks)} chunks de texto gerados prontos para vetorização.")
        
        sample_chunks = chunks[:5] 
        print(f"\n Gerando vetores para uma amostra de {len(sample_chunks)} chunks técnicos...")
        
        embedding_engine = initialize_embedding_model()
        
        texts_to_embed = [doc.page_content for doc in sample_chunks]
        
        start_time = time.time()
        embeddings = embedding_engine.embed_documents(texts_to_embed)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print("-" * 50)
        print(f"Resumo do Pipeline de Embeddings:")
        print(f"   - Chunks processados na amostra: {len(embeddings)}")
        print(f"   - Dimensões de cada vetor gerado: {len(embeddings[0])} eixos numéricos")
        print(f"   - Tempo total de vetorização: {total_time:.2f} segundos")
        print(f"   - Velocidade média: {total_time/len(embeddings):.4f} s/chunk")
        print("-" * 50 + "\n")
        
        print("Amostra dos vetores gerados (Primeiras 5 dimensões de cada direção semântica):\n")
        for i, vector in enumerate(embeddings):
            original_doc = sample_chunks[i]
            source_file = original_doc.metadata.get("source", "Desconhecida")
            
            print(f"--- Vetor {i+1} ---")
            print(f"Documento Pai: {source_file}")
            print(f"Trecho do texto: {original_doc.page_content.strip()[:90]}...")
            print(f"Vetor (Primeiras 5 dimensões): {vector[:5]}")
            print("-" * 30 + "\n")