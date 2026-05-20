import time
from langchain_text_splitters import TokenTextSplitter
from load_document import load_all_documents

def split_documents_by_tokens(documents):

    splitter = TokenTextSplitter(
        chunk_size=150,     # Máximo de tokens por bloco
        chunk_overlap=30     # Sobreposição de tokens entre blocos para não perder o contexto
    )
    chunks = splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    print("Carregando documentos da pasta 'data/raw'...")
    raw_docs = load_all_documents("data/raw")
    
    if not raw_docs:
        print("Nenhum documento disponível. Adicione arquivos de pontes em 'data/raw' para rodar o pipeline.")
    else:
        print("\n Iniciando a divisão dos documentos baseada em TOKENS...")
        
        start_time = time.time()
        chunks = split_documents_by_tokens(raw_docs)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print("-" * 50)
        print(f" Resumo do Pipeline de Token Chunking:")
        print(f"   - Documentos/Páginas de origem: {len(raw_docs)}")
        print(f"   - Total de chunks por tokens gerados: {len(chunks)}")
        print(f"   - Tempo total de processamento: {total_time:.4f} segundos")
        print("-" * 50 + "\n")
        
        print(" Amostra dos chunks de token gerados (Primeiros 3 chunks):\n")
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} ---")
            source_file = chunk.metadata.get("source", "Desconhecida")
            print(f"Origem do Dado: {source_file}")
            print(f"Conteúdo do Bloco:\n{chunk.page_content.strip()}")
            print("-" * 30 + "\n")
            
        if len(chunks) > 3:
            print(f"... e mais {len(chunks) - 3} chunk(s) de token gerado(s) no pipeline.")