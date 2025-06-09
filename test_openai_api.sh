#!/bin/bash
# Bash script to test the OpenAI-compliant RAG API
# Usage: ./test_openai_api.sh

BASE_URL="http://localhost:8000"

echo "Testing OpenAI-compliant RAG API at $BASE_URL"
echo "=================================================="

# Test 1: Health check
echo -e "\n1. Testing health endpoint..."
curl -s "$BASE_URL/health" | jq '.' || echo "Health check failed"

# Test 2: List models
echo -e "\n2. Testing models endpoint..."
curl -s "$BASE_URL/v1/models" | jq '.' || echo "Models endpoint failed"

# Test 3: Chat completions
echo -e "\n3. Testing chat completions endpoint..."
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [
      {
        "role": "user",
        "content": "What is this documentation about?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 150
  }' | jq '.' || echo "Chat completions failed"

# Test 4: Another chat completion
echo -e "\n4. Testing with a specific documentation question..."
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4-turbo",
    "messages": [
      {
        "role": "user",
        "content": "How do I make HTTP requests using this library?"
      }
    ],
    "temperature": 0.5
  }' | jq '.' || echo "Second chat completions failed"

echo -e "\n=================================================="
echo "API testing completed!"
