# RAG Pipeline Setup and Usage Notes

## Quick Reference

### First Time Setup
```bash
# 1. Set up environment variables
cp .env.example .env
# Edit .env with your actual API keys

# 2. Build and start services 
docker-compose up -d --build

# 3. Setup database
docker-compose exec app python main.py setup

# 4. Create symlink to docs (Windows - run PowerShell as Administrator)
New-Item -ItemType SymbolicLink -Path "$(Get-Location)\docs" -Target "C:\Users\garla\git\requests\docs"

# Or use cmd as Administrator
mklink /D docs "C:\Users\garla\git\requests\docs"

# 5. Index documents
docker-compose exec app python main.py index ./docs/
```

### Regular Usage
```bash
# Start services (no rebuild needed unless dependencies changed)
docker-compose up -d --build

# Index new/updated documents
docker-compose exec app python main.py index ./docs/

# Query via CLI
docker-compose exec app python main.py query "Your question here"

# Query with re-ranking
docker-compose exec app python main.py query "Your question here" --rerank

# Web interface: http://localhost:8000
```

## Recent Fixes (June 2025)

### ✅ Mistral Client v1.x Migration
**Problem**: `NotImplementedError: This client is deprecated`

**Fixed**:
- Updated import: `from mistralai import Mistral` (was `from mistralai.client import MistralClient`)
- Updated initialization: `Mistral(api_key=api_key)` (was `MistralClient(api_key=api_key)`)
- Updated embeddings call: `client.embeddings.create(model=model, inputs=[text])` (was `client.embeddings(model=model, input=[text])`)
- Updated dependencies: `mistralai>=1.0.0` ensures latest client

### ✅ Database Setup Issues
**Problem**: `psycopg2.ProgrammingError: vector type not found in the database`

**Fixed**:
- Separated database connection functions
- `get_db_connection()` for basic operations (setup)
- `get_db_connection_with_vector()` for vector operations (index/query)
- Setup creates pgvector extension before trying to register vector type

### ✅ Lazy Client Initialization
**Problem**: API clients initialized even for database-only operations

**Fixed**:
- RAGPipeline only created when needed (index/query commands)
- Setup command no longer requires API keys
- Better separation of concerns

### ✅ Removed Explicit Container Names
**Problem**: `container_name` directives causing conflicts on `docker-compose up`

**Fixed**:
- Removed all `container_name` directives from docker-compose.yml
- Docker Compose now auto-generates unique names (e.g., `rstparse2-app-1`, `rstparse2-db-1`)
- No more naming conflicts when restarting services
- Commands now use service names: `docker-compose exec app` instead of specific container names

## Development Notes

### Dependencies Management
- Using modern `pyproject.toml` with `uv` for fast dependency resolution
- Maintaining `requirements.txt` for compatibility
- Multi-stage Docker builds with dependency caching
- Version pinning for critical packages: `mistralai>=1.0.0`

### Docker Architecture
- **Multi-stage builds**: Separate builder stage for dependencies
- **Volume mounting**: Only mount `./docs` to avoid overwriting installed packages
- **Environment separation**: Different services for CLI app and web interface
- **Network isolation**: Services communicate via Docker network

### Error Handling Improvements
- Better error messages for missing API keys
- Graceful handling of database connection issues
- Validation of file paths and document formats
- Clear separation between setup and runtime errors

## Troubleshooting

### Container Issues
```bash
# Rebuild containers after dependency changes
docker-compose up -d --build

# Check container status
docker-compose ps

# View logs
docker-compose logs app
docker-compose logs db

# Clean rebuild (removes cached layers)
docker-compose down
docker system prune -f
docker-compose up -d --build
```

### Database Issues
```bash
# Reset database (will lose all indexed data)
docker-compose down
docker volume rm rstparse2_postgres_data
docker-compose up -d
docker-compose exec app python main.py setup

# Check database connection
docker-compose exec db psql -U rag_user -d rag_db -c "\dt"

# Check pgvector extension
docker-compose exec db psql -U rag_user -d rag_db -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

### API Key Issues
```bash
# Check environment variables are loaded
docker-compose exec app env | grep -E "(MISTRAL|OPENAI)"

# Test Mistral API key
docker-compose exec app python -c "import os; from mistralai import Mistral; print('Mistral API key:', 'SET' if os.getenv('MISTRAL_API_KEY') else 'NOT SET')"

# Test OpenAI API key
docker-compose exec app python -c "import os; from openai import OpenAI; print('OpenAI API key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

### Performance Tips
- Use `--rerank` for better query results (slower but more accurate)
- Index documents in batches for large document sets
- Monitor PostgreSQL performance with `docker-compose exec db psql -U rag_user -d rag_db -c "SELECT * FROM pg_stat_activity;"`
- Check vector index usage: `docker-compose exec db psql -U rag_user -d rag_db -c "SELECT schemaname, tablename, indexname FROM pg_indexes WHERE tablename = 'documents';"`

## File Structure
```
rstparse2/
├── .env.example          # Template for environment variables
├── .env                  # Your actual API keys (gitignored)
├── docker-compose.yml    # Multi-service setup
├── Dockerfile           # Multi-stage container build
├── pyproject.toml       # Modern Python project config
├── requirements.txt     # Legacy dependency list
├── main.py              # Core RAG pipeline
├── app.py               # FastAPI web interface
├── docs/                # Mounted directory for .rst files
├── README.md            # User documentation
└── notes.md             # This file - developer notes
```

## API Endpoints

### FastAPI Web Server (app.py - port 8000)
- `GET /` - Web chat interface
- `POST /chat` - JSON: `{"message": "question", "rerank": boolean}`
- `GET /health` - Health check

### OpenAI-Compatible API (main.py serve - port 8000)
- `POST /v1/chat/completions` - OpenAI-compatible chat completions endpoint
- `GET /v1/models` - List available models
- `GET /health` - Health check

### Response Formats

#### Web Chat API (/chat)
```json
{
  "response": "Answer from the RAG pipeline",
  "error": null
}
```

#### OpenAI-Compatible API (/v1/chat/completions)
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1672531200,
  "model": "gpt-4-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Answer from the RAG pipeline"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 75,
    "total_tokens": 225
  }
}
```

### Usage Examples

#### Using the OpenAI-Compatible Endpoint
```bash
# Test the OpenAI-compatible endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [
      {"role": "user", "content": "How do I configure the database?"}
    ],
    "temperature": 0.7
  }'

# List available models
curl http://localhost:8000/v1/models
```

#### Using with OpenAI Python Client
```python
from openai import OpenAI

# Point to your local RAG server
client = OpenAI(
    api_key="dummy-key",  # Not used but required by client
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "user", "content": "How do I configure the database?"}
    ]
)

print(response.choices[0].message.content)
```

## Next Steps / TODOs

- [ ] Add document chunking for large files
- [ ] Implement semantic caching for repeated queries
- [ ] Add support for multiple document formats (PDF, Markdown)
- [ ] Implement user authentication for web interface
- [ ] Add metrics and monitoring (query latency, accuracy)
- [ ] Optimize vector search performance
- [ ] Add batch indexing capabilities
- [ ] Implement document versioning and updates
