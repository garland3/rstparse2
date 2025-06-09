import os
import re
import argparse
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from pgvector import Vector
from docutils.core import publish_parts
from docutils.parsers.rst import directives, roles
from docutils.parsers.rst.directives import unchanged
from docutils import nodes
from docutils.parsers.rst import Directive
from mistralai import Mistral
from openai import OpenAI
import yake
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
import uuid

# --- Configuration ---
# Load from environment variables with defaults
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "rag_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "rag_password")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

EMBEDDING_MODEL = os.getenv("MISTRAL_MODEL", "mistral-embed")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

# Chunking configuration
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "6000"))
CHUNK_THRESHOLD_TOKENS = int(os.getenv("CHUNK_THRESHOLD_TOKENS", "7000"))

# --- Pydantic Models for OpenAI API Compatibility ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage

# --- Database Helper Functions ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    return conn

def get_db_connection_with_vector():
    """Establishes a connection to the PostgreSQL database with vector support."""
    conn = get_db_connection()
    register_vector(conn)
    return conn

def setup_database():
    """Sets up the necessary tables in the database."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    file_path TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    embedding VECTOR(1024)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS functions (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER REFERENCES documents(id),
                    function_signature TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS keywords (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER REFERENCES documents(id),
                    keyword TEXT NOT NULL
                );
            """)
    print("Database setup complete.")

# --- Custom Directive Handlers for Sphinx Compatibility ---
class IgnoredDirective(Directive):
    """A directive that ignores its content but extracts useful text."""
    has_content = True
    required_arguments = 0
    optional_arguments = 10
    final_argument_whitespace = True
    option_spec = {
        'members': unchanged,
        'undoc-members': unchanged,
        'show-inheritance': unchanged,
        'inherited-members': unchanged,
        'maxdepth': unchanged,
    }

    def run(self):
        # Extract the directive name and arguments for potential text content
        directive_name = self.name
        arguments = ' '.join(self.arguments) if self.arguments else ''
        content = '\n'.join(self.content) if self.content else ''
        
        # Create a simple paragraph with extracted text
        text_content = f"{directive_name} {arguments} {content}".strip()
        if text_content:
            paragraph = nodes.paragraph(text=text_content)
            return [paragraph]
        return []

def custom_role_handler(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """Custom role handler that extracts text content from Sphinx roles."""
    # Create a simple text node with the role content
    node = nodes.Text(text)
    return [node], []

def register_sphinx_directives():
    """Register custom handlers for common Sphinx directives and roles."""
    # Register directive handlers
    sphinx_directives = [
        'automodule', 'autoclass', 'autofunction', 'automethod', 'autodata',
        'autoexception', 'autoattribute', 'module', 'currentmodule',
        'toctree', 'versionadded', 'versionchanged', 'deprecated',
        'seealso', 'note', 'warning', 'todo', 'highlight'
    ]
    
    for directive_name in sphinx_directives:
        directives.register_directive(directive_name, IgnoredDirective)
    
    # Register role handlers for common Sphinx roles
    sphinx_roles = [
        'class', 'func', 'meth', 'mod', 'obj', 'data', 'const', 'attr',
        'exc', 'ref', 'doc', 'download', 'numref', 'eq', 'term'
    ]
    
    for role_name in sphinx_roles:
        roles.register_local_role(role_name, custom_role_handler)

# --- Helper Functions ---
def parse_rst(file_path):
    """Parses an .rst file and extracts the text content."""
    # Register custom directive handlers before parsing
    register_sphinx_directives()
    
    with open(file_path, "r", encoding="utf-8") as file:
        rst_content = file.read()
    
    # Handle common Sphinx substitutions
    rst_content = rst_content.replace('|version|', '1.0.0')
    rst_content = rst_content.replace('|release|', '1.0.0')
    
    # Fix include directive paths that are absolute
    rst_content = re.sub(r'\.\. include:: /([^/\n]+)', r'.. include:: \1', rst_content)
    
    # Remove problematic include directives that reference missing files
    rst_content = re.sub(r'\.\. include:: [^\n]*HISTORY\.md[^\n]*\n?', '', rst_content)
    rst_content = re.sub(r'\.\. include:: [^\n]*\.\.\/[^\n]*\n?', '', rst_content)
    
    try:
        parts = publish_parts(source=rst_content, writer_name="html")
        # A simple way to clean up the HTML a bit
        text = re.sub('<[^<]+?>', '', parts['html_body'])
        return text
    except Exception as e:
        print(f"Warning: Could not fully parse {file_path}: {e}")
        # Fallback: return raw text with basic cleanup
        text = re.sub(r'\.\. [a-zA-Z-]+::[^\n]*\n', '', rst_content)  # Remove directive lines
        text = re.sub(r':[a-zA-Z-]+:`[^`]*`', '', text)  # Remove roles
        return text

def extract_functions(text):
    """Extracts function definitions from the text."""
    # This regex is for .. function:: directive in Sphinx
    function_pattern = r"\.\.\s+function::\s+(.*)"
    functions = re.findall(function_pattern, text)
    return functions

def extract_keywords_from_text(text, num_keywords=10):
    """Extracts keywords from the text using YAKE."""
    kw_extractor = yake.KeywordExtractor(n=1, top=num_keywords, features=None)
    keywords = [kw for kw, _ in kw_extractor.extract_keywords(text)]
    return keywords

def chunk_text(text, max_tokens=None):
    """Split text into chunks that fit within token limits."""
    if max_tokens is None:
        max_tokens = MAX_CHUNK_TOKENS
    
    # Simple chunking by sentences to avoid breaking mid-sentence
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # More conservative token estimation: ~3 characters per token
        estimated_tokens = len(current_chunk + sentence) // 3
        
        if estimated_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
        else:
            current_chunk += sentence + ". "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# --- RAG Pipeline ---
class RAGPipeline:
    def __init__(self):
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    def get_embedding(self, text):
        """Generates an embedding for the given text using Mistral."""
        embeddings_batch_response = self.mistral_client.embeddings.create(
            model=EMBEDDING_MODEL,
            inputs=[text]
        )
        return embeddings_batch_response.data[0].embedding

    def index_documents(self, doc_paths):
        """Parses, vectorizes, and indexes the Sphinx documents."""
        with get_db_connection_with_vector() as conn:
            with conn.cursor() as cur:
                total_chunks = 0
                for doc_path in doc_paths:
                    print(f"Indexing {doc_path}...")
                    content = parse_rst(doc_path)
                    
                    # Check if content is too long and needs chunking
                    estimated_tokens = len(content) // 4
                    if estimated_tokens > CHUNK_THRESHOLD_TOKENS:
                        print(f"  Document is large ({estimated_tokens} estimated tokens), chunking...")
                        chunks = chunk_text(content)
                        print(f"  Split into {len(chunks)} chunks")
                        
                        for i, chunk in enumerate(chunks):
                            chunk_path = f"{doc_path}#chunk{i+1}"
                            embedding = self.get_embedding(chunk)
                            
                            # Insert document chunk and get its ID
                            cur.execute(
                                "INSERT INTO documents (file_path, content, embedding) VALUES (%s, %s, %s) ON CONFLICT (file_path) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding RETURNING id;",
                                (chunk_path, chunk, embedding)
                            )
                            doc_id = cur.fetchone()[0]

                            # Extract and store functions from chunk
                            functions = extract_functions(chunk)
                            if functions:
                                execute_values(cur, "INSERT INTO functions (doc_id, function_signature) VALUES %s", [(doc_id, f) for f in functions])
                            
                            # Extract and store keywords from chunk
                            keywords = extract_keywords_from_text(chunk)
                            if keywords:
                                execute_values(cur, "INSERT INTO keywords (doc_id, keyword) VALUES %s", [(doc_id, k) for k in keywords])
                        
                        total_chunks += len(chunks)
                    else:
                        # Document is small enough, index as-is
                        embedding = self.get_embedding(content)
                        
                        # Insert document and get its ID
                        cur.execute(
                            "INSERT INTO documents (file_path, content, embedding) VALUES (%s, %s, %s) ON CONFLICT (file_path) DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding RETURNING id;",
                            (doc_path, content, embedding)
                        )
                        doc_id = cur.fetchone()[0]

                        # Extract and store functions
                        functions = extract_functions(content)
                        if functions:
                            execute_values(cur, "INSERT INTO functions (doc_id, function_signature) VALUES %s", [(doc_id, f) for f in functions])
                        
                        # Extract and store keywords
                        keywords = extract_keywords_from_text(content)
                        if keywords:
                            execute_values(cur, "INSERT INTO keywords (doc_id, keyword) VALUES %s", [(doc_id, k) for k in keywords])
                        
                        total_chunks += 1

        print(f"Indexed {len(doc_paths)} documents ({total_chunks} total chunks).")

    def search(self, query, top_k=5):
        """Performs vector search and keyword search."""
        query_embedding = self.get_embedding(query)
        
        with get_db_connection_with_vector() as conn:
            with conn.cursor() as cur:
                # Vector search - format embedding as string for pgvector
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                cur.execute(
                    "SELECT content FROM documents ORDER BY embedding <=> %s LIMIT %s",
                    (embedding_str, top_k)
                )
                vector_results = [row[0] for row in cur.fetchall()]

                # Keyword search (simple version)
                query_keywords = query.split()
                cur.execute(
                    "SELECT content FROM documents d JOIN keywords k ON d.id = k.doc_id WHERE k.keyword = ANY(%s) LIMIT %s",
                    (query_keywords, top_k)
                )
                keyword_results = [row[0] for row in cur.fetchall()]

        # Combine and deduplicate results
        combined_results = list(dict.fromkeys(vector_results + keyword_results))
        return combined_results

    def answer_question(self, query, rerank=False):
        """Answers a question using the RAG pipeline."""
        search_results = self.search(query)

        if not search_results:
            print("No relevant documents found.")
            return

        context_docs = search_results
        if rerank:
            print("Re-ranking results...")
            # Simple re-ranking based on query keyword overlap
            try:
                query_words = set(query.lower().split())
                scored_results = []
                
                for doc in search_results:
                    doc_words = set(doc.lower().split())
                    overlap_score = len(query_words.intersection(doc_words))
                    scored_results.append((overlap_score, doc))
                
                # Sort by overlap score (descending) and take the documents
                context_docs = [doc for score, doc in sorted(scored_results, key=lambda x: x[0], reverse=True)]
                print("Results re-ranked based on keyword overlap.")
            except Exception as e:
                print(f"Could not perform re-ranking: {e}. Using initial search results.")
                context_docs = search_results
        
        context = "\n\n---\n\n".join(context_docs)
        
        prompt = f"""
        Based on the following context, please answer the question.

        Context:
        {context}

        Question: {query}

        Answer:
        """

        print("\n--- Sending to LLM ---")
        response = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documentation."},
                {"role": "user", "content": prompt}
            ]
        )
        print("\n--- LLM Response ---")
        print(response.choices[0].message.content)
    
    def get_rag_response(self, messages, model=None, temperature=0.7, max_tokens=None):
        """Get a RAG-enhanced response for chat completions API."""
        if not messages:
            raise ValueError("No messages provided")
        
        # Extract the last user message as the query
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            raise ValueError("No user messages found")
        
        query = user_messages[-1].content
        
        # Get relevant context using search
        search_results = self.search(query)
        
        if search_results:
            context = "\n\n---\n\n".join(search_results)
            
            # Build the enhanced messages with context
            enhanced_messages = []
            
            # Add system message with context
            system_content = f"""You are a helpful assistant that answers questions based on provided documentation. Use the following context to answer questions accurately.

Context:
{context}

Instructions:
- Answer based on the provided context
- If the context doesn't contain relevant information, say so
- Be concise and accurate
- Cite specific information from the context when possible"""
            
            enhanced_messages.append({"role": "system", "content": system_content})
            
            # Add the conversation history
            for msg in messages:
                enhanced_messages.append({"role": msg.role, "content": msg.content})
        else:
            # No context found, use original messages with a basic system prompt
            enhanced_messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for msg in messages:
                enhanced_messages.append({"role": msg.role, "content": msg.content})
        
        # Call the LLM
        response = self.openai_client.chat.completions.create(
            model=model or OPENAI_MODEL,
            messages=enhanced_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response

# --- FastAPI Application ---
app = FastAPI(title="RAG Chat Completions API", version="1.0.0")

# Global RAG pipeline instance
rag_pipeline = None

def get_rag_pipeline():
    """Get or create the RAG pipeline instance."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with RAG enhancement."""
    try:
        pipeline = get_rag_pipeline()
        
        # Get RAG-enhanced response
        response = pipeline.get_rag_response(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Convert to our response format
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response.choices[0].message.content
                    ),
                    finish_reason=response.choices[0].finish_reason or "stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0
            )
        )
        
        return chat_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible endpoint)."""
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "rag-system"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": int(time.time())}

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="A RAG pipeline for Sphinx documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `setup` command
    parser_setup = subparsers.add_parser("setup", help="Set up the database.")
    
    # `index` command
    parser_index = subparsers.add_parser("index", help="Index .rst documents.")
    parser_index.add_argument("paths", nargs="+", help="Paths to the .rst files or directories.")

    # `query` command
    parser_query = subparsers.add_parser("query", help="Ask a question.")
    parser_query.add_argument("question", type=str, help="The question to ask.")
    parser_query.add_argument("--rerank", action="store_true", help="Enable re-ranking of search results.")

    # `serve` command
    parser_serve = subparsers.add_parser("serve", help="Start the FastAPI server for chat completions.")
    parser_serve.add_argument("--host", default="0.0.0.0", help="Host to bind the server to.")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to bind the server to.")

    args = parser.parse_args()

    if args.command == "setup":
        setup_database()
    elif args.command == "index":
        doc_paths = []
        for path in args.paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.endswith(".rst"):
                            doc_paths.append(os.path.join(root, file))
            elif os.path.isfile(path) and path.endswith(".rst"):
                doc_paths.append(path)
        
        if not doc_paths:
            print("No .rst files found in the specified paths.")
            return
            
        rag_pipeline = RAGPipeline()
        rag_pipeline.index_documents(doc_paths)
    elif args.command == "query":
        rag_pipeline = RAGPipeline()
        rag_pipeline.answer_question(args.question, rerank=args.rerank)
    elif args.command == "serve":
        print(f"Starting RAG Chat Completions API server on {args.host}:{args.port}")
        print("Available endpoints:")
        print(f"  - POST http://{args.host}:{args.port}/v1/chat/completions")
        print(f"  - GET  http://{args.host}:{args.port}/v1/models")
        print(f"  - GET  http://{args.host}:{args.port}/health")
        uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
