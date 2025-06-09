from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
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
        # Capture the response from the RAG pipeline
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        # Capture stdout to get the LLM response
        captured_output = io.StringIO()
        
        with redirect_stdout(captured_output):
            rag_pipeline.answer_question(request.message, rerank=request.rerank)
        
        output = captured_output.getvalue()
        
        # Extract the LLM response from the captured output
        if "--- LLM Response ---" in output:
            response_text = output.split("--- LLM Response ---")[-1].strip()
        else:
            response_text = "No response generated. Please check if documents are indexed."
        
        return ChatResponse(response=response_text)
    
    except Exception as e:
        return ChatResponse(response="", error=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chat Interface</title>
        <script src="/static/js/marked.min.js"></script>
        <script src="/static/js/prism-core.min.js"></script>
        <script src="/static/js/prism-autoloader.min.js"></script>
        <link href="/static/css/prism.css" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .chat-container {
                width: 90%;
                height: 90vh;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            
            .chat-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                text-align: center;
            }
            
            .chat-header h1 {
                font-size: 24px;
                margin-bottom: 5px;
            }
            
            .chat-header p {
                opacity: 0.9;
                font-size: 14px;
            }
            
            .chat-messages {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                background: #f8f9fa;
            }
            
            .message {
                margin-bottom: 15px;
                display: flex;
                align-items: flex-start;
            }
            
            .message.user {
                justify-content: flex-end;
            }
            
            .message-content {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 18px;
                word-wrap: break-word;
            }
            
            .message.user .message-content {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .message.bot .message-content {
                background: white;
                border: 1px solid #e1e5e9;
                color: #333;
            }
            
            .message.error .message-content {
                background: #fee;
                border: 1px solid #fcc;
                color: #c33;
            }
            
            .chat-input-container {
                padding: 20px;
                background: white;
                border-top: 1px solid #e1e5e9;
            }
            
            .chat-input-form {
                display: flex;
                gap: 10px;
                align-items: center;
            }
            
            .chat-input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e1e5e9;
                border-radius: 25px;
                font-size: 16px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            .chat-input:focus {
                border-color: #667eea;
            }
            
            .send-button {
                padding: 12px 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                transition: transform 0.2s;
            }
            
            .send-button:hover {
                transform: translateY(-2px);
            }
            
            .send-button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .options {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-top: 10px;
            }
            
            .checkbox-container {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Markdown styling */
            .message-content h1, .message-content h2, .message-content h3,
            .message-content h4, .message-content h5, .message-content h6 {
                margin: 10px 0 5px 0;
                font-weight: bold;
            }
            
            .message-content h1 { font-size: 1.5em; }
            .message-content h2 { font-size: 1.3em; }
            .message-content h3 { font-size: 1.1em; }
            
            .message-content p {
                margin: 8px 0;
                line-height: 1.4;
            }
            
            .message-content ul, .message-content ol {
                margin: 8px 0;
                padding-left: 20px;
            }
            
            .message-content li {
                margin: 4px 0;
            }
            
            .message-content blockquote {
                border-left: 4px solid #667eea;
                margin: 10px 0;
                padding: 5px 0 5px 15px;
                background: #f8f9fa;
                font-style: italic;
            }
            
            .message-content code {
                background: #f1f3f4;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.9em;
            }
            
            .message-content pre {
                background: #f8f8f8;
                border: 1px solid #e1e5e9;
                border-radius: 6px;
                padding: 12px;
                margin: 10px 0;
                overflow-x: auto;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.9em;
                line-height: 1.4;
            }
            
            .message-content pre code {
                background: none;
                padding: 0;
                border-radius: 0;
                font-size: inherit;
            }
            
            .message-content table {
                border-collapse: collapse;
                margin: 10px 0;
                width: 100%;
            }
            
            .message-content th, .message-content td {
                border: 1px solid #e1e5e9;
                padding: 8px 12px;
                text-align: left;
            }
            
            .message-content th {
                background: #f8f9fa;
                font-weight: bold;
            }
            
            .message-content a {
                color: #667eea;
                text-decoration: none;
            }
            
            .message-content a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>RAG Chat Interface</h1>
                <p>Ask questions about your indexed Sphinx documentation</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message bot">
                    <div class="message-content">
                        Hello! I'm here to help you find information from your indexed documentation. Ask me anything!
                    </div>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                Processing your question...
            </div>
            
            <div class="chat-input-container">
                <form class="chat-input-form" id="chatForm">
                    <input 
                        type="text" 
                        class="chat-input" 
                        id="messageInput" 
                        placeholder="Type your question here..." 
                        required
                    >
                    <button type="submit" class="send-button" id="sendButton">Send</button>
                </form>
                <div class="options">
                    <div class="checkbox-container">
                        <input type="checkbox" id="rerankCheckbox">
                        <label for="rerankCheckbox">Enable re-ranking</label>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const chatMessages = document.getElementById('chatMessages');
            const chatForm = document.getElementById('chatForm');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const rerankCheckbox = document.getElementById('rerankCheckbox');
            const loading = document.getElementById('loading');

            function addMessage(content, isUser = false, isError = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : (isError ? 'error' : 'bot')}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                if (isUser || isError) {
                    // For user messages and errors, use plain text
                    contentDiv.textContent = content;
                } else {
                    // For bot messages, render markdown
                    try {
                        const htmlContent = marked.parse(content);
                        contentDiv.innerHTML = htmlContent;
                        
                        // Highlight code blocks
                        contentDiv.querySelectorAll('pre code').forEach((block) => {
                            Prism.highlightElement(block);
                        });
                    } catch (error) {
                        // Fallback to plain text if markdown parsing fails
                        contentDiv.textContent = content;
                    }
                }
                
                messageDiv.appendChild(contentDiv);
                chatMessages.appendChild(messageDiv);
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            function setLoading(isLoading) {
                loading.classList.toggle('show', isLoading);
                sendButton.disabled = isLoading;
                messageInput.disabled = isLoading;
            }

            chatForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const message = messageInput.value.trim();
                if (!message) return;
                
                // Add user message
                addMessage(message, true);
                
                // Clear input
                messageInput.value = '';
                
                // Show loading
                setLoading(true);
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            rerank: rerankCheckbox.checked
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        addMessage(`Error: ${data.error}`, false, true);
                    } else {
                        addMessage(data.response || 'No response received.');
                    }
                } catch (error) {
                    addMessage(`Error: ${error.message}`, false, true);
                } finally {
                    setLoading(false);
                    messageInput.focus();
                }
            });

            // Focus input on load
            messageInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
