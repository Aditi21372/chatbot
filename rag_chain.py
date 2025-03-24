import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import LLMResult
from typing import List
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class FreeLLM(LLM):
    """Custom LLM wrapper for Hugging Face TinyLlama, integrating with LangChain."""
    
    gen_func: callable  # Generation function

    def _generate(self, prompts: List[str], stop=None) -> LLMResult:
        """Generates structured responses for multiple prompts."""
        responses = [{
            "text": self.gen_func(
                prompt,
                max_new_tokens=100,
                temperature=0.7  # Removed truncation here since it's hardcoded in generate_free_response
            )
        } for prompt in prompts]
        return LLMResult(generations=[[response] for response in responses])

    def _call(self, prompt: str, stop=None) -> str:
        """Generates a response for a single prompt."""
        return self.gen_func(
            prompt,
            max_new_tokens=100,
            temperature=0.7  # Removed truncation here as well
        )

    @property
    def _llm_type(self) -> str:
        """Defines the type of the LLM for LangChain."""
        return "custom_free_llm"

    class Config:
        arbitrary_types_allowed = True  # Fix for Pydantic v2 compatibility

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()

    def get_llm(self):
        """Chooses between OpenAI GPT (if API key is set) or FreeLLM."""
        if os.getenv("OPENAI_API_KEY"):
            from langchain.llms import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        else:
            from free_llm import generate_free_response
            # Simply forward the prompt and kwargs without setting defaults here.
            return FreeLLM(gen_func=lambda prompt, **kwargs: generate_free_response(prompt, **kwargs))

    def create_chain(self):
        """Creates a retrieval-based chatbot using FAISS."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

        # Improved prompt for concise, relevant answers
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Answer the following question using only the provided context. Be brief and informative. "
                "If you don't know the answer, say 'I don't know.'\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template}
        )

        return qa_chain

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer

    print("🔍 Loading and processing the document...")
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()
    print(f"✅ Texts loaded and split! {len(texts)} chunks found.")

    print("🔍 Creating FAISS vector store...")
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    print("✅ FAISS vector store created!")

    print("🔍 Creating RAG chatbot chain...")
    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()
    print("✅ RAG chatbot chain created!")

    query = "strategies for social interaction"
    print(f"🔍 Debug: Query - {query}")

    print("🔍 Running chatbot on query...")
    try:
        result = qa_chain.invoke({"query": query})
        print(f"✅ Answer: {result['result']}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
