# We will create a class that uses the Hugging Face Sentence Transformers library to create embeddings for text data. We will then use the embeddings to create a FAISS vector store.
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Correct import
from langchain_community.vectorstores import FAISS


class EmbeddingIndexer:
    def __init__(self):
        # Using a free Hugging Face embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def create_vectorstore(self, texts):
        """Creates a FAISS vector store from document chunks."""
        vectorstore = FAISS.from_documents(texts, self.embeddings)
        return vectorstore

if __name__ == "__main__":
    from document_processor import DocumentProcessor

    # Load and split text
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    # Create the FAISS vector store
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    print("✅ FAISS vector store created successfully!")
