from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import your RAGChain and any other necessary modules
from rag_chain import RAGChain
from embedding_indexer import EmbeddingIndexer
from document_processor import DocumentProcessor

# Initialize FastAPI app
app = FastAPI(title="Neurodivergent-Friendly Chatbot API")

# Create a Pydantic model for chat requests
class ChatRequest(BaseModel):
    query: str

# Global variable to hold the chatbot chain
qa_chain = None

def initialize_chatbot():
    global qa_chain
    # Load and process the document (adjust path as needed)
    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()
    # Create the FAISS vector store
    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)
    # Create the RAG chain
    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()

# Initialize chatbot at startup
initialize_chatbot()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
        except WebSocketDisconnect:
            print("WebSocket connection closed")
            break
        if qa_chain is None:
            await websocket.send_text("Chatbot is not initialized.")
            continue
        try:
            result = qa_chain.invoke({"query": data})
            await websocket.send_text(result["result"])
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if qa_chain is None:
        raise HTTPException(status_code=500, detail="Chatbot is not initialized.")
    try:
        result = qa_chain.invoke({"query": request.query})
        return {"query": request.query, "result": result["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
