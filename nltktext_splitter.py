import time
import nltk
from langchain_text_splitters import NLTKTextSplitter
from load_document import load_all_documents

def download_nltk_dependencies():
    resources = ["tokenizers/punkt", "tokenizers/punkt_tab"]
    
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            resource_name = resource.split("/")[-1]
            print(f"📥 Recurso '{resource_name}' do NLTK não encontrado. Baixando...")
            nltk.download(resource_name, quiet=True)

def split_documents_by_nltk(documents):

    download_nltk_dependencies()
    text_splitter = NLTKTextSplitter(
        chunk_size=600,       # Max caracteres por bloco (baseado em sentenças completas)
        chunk_overlap=100,    # Sobreposição de segurança para preservar a continuidade
        language="portuguese" 
    )
    
    split_docs = text_splitter.split_documents(documents)
    return split_docs

if __name__ == "__main__":
    print("Carregando documentos da pasta 'data/raw'...")
    raw_docs = load_all_documents("data/raw")
    
    if not raw_docs:
        print("Nenhum documento disponível. Adicione a literatura de pontes em 'data/raw' para testar.")
    else:
        print("\n Iniciando a divisão gramatical baseada em sentenças (NLTK)...")
        
        start_time = time.time()
        split_docs = split_documents_by_nltk(raw_docs)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        print("-" * 50)
        print(f"Resumo do Pipeline NLTK Splitter:")
        print(f"   - Documentos/Páginas originais: {len(raw_docs)}")
        print(f"   - Total de chunks gramaticais gerados: {len(split_docs)}")
        print(f"   - Tempo total de processamento: {total_time:.4f} segundos")
        print("-" * 50 + "\n")
        
        print("Amostra dos chunks gerados pelo NLTK (Primeiros 3 chunks):\n")
        for i, chunk in enumerate(split_docs[:3]):
            print(f"--- Chunk {i+1} ---")
            source_file = chunk.metadata.get("source", "Desconhecida")
            print(f"Origem do Dado: {source_file}")
            print(f"Conteúdo do Bloco:\n{chunk.page_content.strip()}")
            print("-" * 30 + "\n")
            
        if len(split_docs) > 3:
            print(f"... e mais {len(split_docs) - 3} chunk(s) gerado(s) via PLN.")