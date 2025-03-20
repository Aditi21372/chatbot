# document_processor.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        """Loads a text document and splits it into smaller chunks."""
        # Load the document from a text file
        loader = TextLoader(self.file_path)
        documents = loader.load()

        # Split the document into chunks of 1000 characters (with overlap for context)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        return texts

if __name__ == "__main__":
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()
    print(f"✅ Processed {len(texts)} text chunks")
