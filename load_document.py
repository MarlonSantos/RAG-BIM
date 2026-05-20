
import os
import time
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader

def load_all_documents(data_dir="data/raw"):

    all_docs_list = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Pasta '{data_dir}' não encontrada. Ela já foi criada automaticamente, porém está vazia.")
        print(f"Insira seus arquivos de documentação BIM (PDF, TXT ou DOCX) nela para rodar o pipeline.\n")
        return all_docs_list
    

    files = os.listdir(data_dir)
    if not files:
        print(f"A pasta '{data_dir}' está vazia. Adicione arquivos para processar.")
        return all_docs_list

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.isdir(file_path):
                continue
        try:
            if file_name.endswith(".pdf"):
                print(f"Carregando PDF: {file_name}")
                loader = PyPDFLoader(file_path)
                all_docs_list.extend(loader.load())
                
            elif file_name.endswith(".txt"):
                print(f"Carregando TXT: {file_name}")
                loader = TextLoader(file_path, encoding="utf-8")
                all_docs_list.extend(loader.load())
                
            elif file_name.endswith(".docx"):
                print(f"Carregando DOCX: {file_name}")
                loader = UnstructuredWordDocumentLoader(file_path)
                all_docs_list.extend(loader.load())     
        except Exception as e:
            print(f"Erro ao carregar o arquivo {file_name}: {str(e)}")
    return all_docs_list
#=======================================================================================================================================
if __name__ == "__main__":
    print("Carregando documentos...")
    start_time = time.time()
    docs = load_all_documents("data/raw")
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total de páginas/documentos carregados: {len(docs)}\n")
    print(f"Tempo total de execução: {total_time:.2f} segundos")
    
    for i, doc in enumerate(docs):  
        print(f"--- Documento {i+1} ---")
        source = doc.metadata.get("source", "Desconhecida")
        print(f"Origem: {source}")
        print(f"Conteúdo: {doc.page_content[:275]}...")
        print("\n")