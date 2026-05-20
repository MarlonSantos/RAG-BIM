import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_document import load_all_documents

def split_processed_documents(documents):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Máximo de caracteres por chunk
        chunk_overlap=100,     # Sobra de caracteres para preservar contexto entre blocos
        separators=["\n\n", "\n", ".", " ", ""], # Critérios de quebra ordenados
    )
    
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    print("Carregando documentos da pasta 'data/raw'...")
    raw_docs = load_all_documents("data/raw")
    
    if not raw_docs:
        print("Nenhum documento disponível para divisão. Adicione arquivos em 'data/raw'.")
    else:
        print("\nIniciando a divisão dos documentos em chunks...")
        
        start_time = time.time()
        chunks = split_processed_documents(raw_docs)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print("-" * 50)
        print(f" Resumo do Pipeline de Chunking:")
        print(f"   - Documentos/Páginas de origem: {len(raw_docs)}")
        print(f"   - Total de chunks gerados: {len(chunks)}")
        print(f"   - Tempo total de processamento: {total_time:.4f} segundos")
        print("-" * 50 + "\n")
        
        print("Amostra dos chunks gerados (Primeiros 3 chunks):\n")
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} ({len(chunk.page_content)} caracteres) ---")
            source_file = chunk.metadata.get("source", "Desconhecida")
            print(f"Origem do Dado: {source_file}")
            print(f"Conteúdo do Bloco:\n{chunk.page_content.strip()}")
            print("-" * 30 + "\n")
            
        if len(chunks) > 3:
            print(f"... e mais {len(chunks) - 3} chunk(s) gerado(s) no pipeline.")