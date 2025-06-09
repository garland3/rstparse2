from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import time
import uuid
from main import RAGPipeline

app = FastAPI(title="RAG Chat Interface", description="Chat interface for Sphinx documentation RAG pipeline")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

class ChatRequest(BaseModel):
    message: str
    rerank: bool = False

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    error: str = None

# OpenAI-compatible models
class OpenAIMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str

class OpenAIModelsResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIModel]

# OpenAI-compatible API endpoints
@app.get("/v1/models", response_model=OpenAIModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    models = [
        OpenAIModel(
            id="gpt-4-turbo",
            created=int(time.time()),
            owned_by="rag-system"
        ),
        OpenAIModel(
            id="gpt-3.5-turbo",
            created=int(time.time()),
            owned_by="rag-system"
        )
    ]
    return OpenAIModelsResponse(data=models)

@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def chat_completions(request: OpenAIChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        # Extract the user message from the messages array
        user_message = ""
        for message in request.messages:
            if message.role == "user":
                user_message = message.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")
        
        # Capture the response from the RAG pipeline
        import io
        from contextlib import redirect_stdout
        
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            rag_pipeline.answer_question(user_message, rerank=False)
        
        output = captured_output.getvalue()
        
        # Extract the LLM response from the captured output
        if "--- LLM Response ---" in output:
            response_text = output.split("--- LLM Response ---")[-1].strip()
        else:
            response_text = "No response generated. Please check if documents are indexed."
        
        # Create OpenAI-compatible response
        response_id = f"chatcmpl-{str(uuid.uuid4())}"
        created_time = int(time.time())
        
        # Estimate token usage (rough approximation)
        prompt_tokens = len(user_message.split()) * 1.3  # Rough estimate
        completion_tokens = len(response_text.split()) * 1.3
        total_tokens = int(prompt_tokens + completion_tokens)
        
        choice = OpenAIChoice(
            index=0,
            message=OpenAIMessage(role="assistant", content=response_text),
            finish_reason="stop"
        )
        
        usage = OpenAIUsage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total_tokens
        )
        
        return OpenAIChatResponse(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[choice],
            usage=usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the RAG pipeline and get a response.
    """
    try:
        # Use the new method that returns both response and sources
        response_text, sources = rag_pipeline.get_answer_with_sources(request.message, rerank=request.rerank)
        
        return ChatResponse(response=response_text, sources=sources)
    
    except Exception as e:
        return ChatResponse(response="", error=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/")
async def get_chat_interface():
    """Serve the chat interface."""
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
