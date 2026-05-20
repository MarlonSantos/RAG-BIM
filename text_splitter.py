import time
from langchain_text_splitters import CharacterTextSplitter
from load_document import load_all_documents

def split_documents_by_exact_character(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",        # Ponto exato e obrigatório de corte
        chunk_size=400,       # Máximo de caracteres por bloco
        chunk_overlap=50,      # Sobreposição de caracteres para manter a continuidade
        length_function=len    # Função padrão de medição (caracteres)
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

if __name__ == "__main__":
    print("Carregando documentos da pasta 'data/raw'...")
    raw_docs = load_all_documents("data/raw")
    
    if not raw_docs:
        print("Nenhum documento disponível para divisão. Insira seus arquivos normativos em 'data/raw'.")
    else:
        print("\n Iniciando a divisão determinística por quebra de linha (\\n)...")
        
        start_time = time.time()
        split_docs = split_documents_by_exact_character(raw_docs)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print("-" * 50)
        print(f"Resumo do Pipeline de Character Splitter:")
        print(f"   - Documentos/Páginas originais: {len(raw_docs)}")
        print(f"   - Total de chunks determinísticos criados: {len(split_docs)}")
        print(f"   - Tempo total de processamento: {total_time:.4f} segundos")
        print("-" * 50 + "\n")
        
        print("Amostra dos chunks gerados (Primeiros 3 chunks):\n")
        for i, doc in enumerate(split_docs[:3]):
            print(f"--- Chunk {i+1} ---")
            source_file = doc.metadata.get("source", "Desconhecida")
            print(f"Origem do Dado: {source_file}")
            print(f"Conteúdo do Bloco:\n{doc.page_content.strip()}")
            print("-" * 30 + "\n")
            
        if len(split_docs) > 3:
            print(f"... e mais {len(split_docs) - 3} chunk(s) gerado(s) de forma rígida.")