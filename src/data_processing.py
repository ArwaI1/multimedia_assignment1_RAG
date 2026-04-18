import os
from byaldi import RAGMultiModalModel

def setup_rag_index(pdf_path="data/raw/9789240049338-eng.pdf", index_name="stress_health_book_index"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: Cannot find the PDF at {pdf_path}. Please make sure it is in data/raw/")

    index_folder_path = f".byaldi/{index_name}"

    if os.path.exists(index_folder_path):
        print(f"Index '{index_name}' found! Loading instantly on CPU...")
        RAG = RAGMultiModalModel.from_index(index_name, device="cpu")
    else:
        print(f"No index found. Preparing to load the AI model to create '{index_name}'...")
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device="cpu")
        
        print("\n" + "="*50)
        print("AI Model Loaded Successfully!")
        print("NOW BEGINNING INDEXING: Converting PDF to images and reading pages.")
        print("This might take 15-30 minutes on a CPU. Please do not close the terminal!")
        print("="*50 + "\n")
        
        RAG.index(
            input_path=pdf_path,
            index_name=index_name,
            store_collection_with_index=True,
            overwrite=True
        )
        print("Indexing complete and saved.")
    
    return RAG