# PowerShell script to test the OpenAI-compliant RAG API
# Usage: .\test_openai_api.ps1

$baseUrl = "http://localhost:8000"

Write-Host "Testing OpenAI-compliant RAG API at $baseUrl" -ForegroundColor Green
Write-Host "=" * 50

# Test 1: Health check
Write-Host "`n1. Testing health endpoint..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-RestMethod -Uri "$baseUrl/health" -Method GET
    Write-Host "✓ Health check passed:" -ForegroundColor Green
    $healthResponse | ConvertTo-Json -Depth 3
} catch {
    Write-Host "✗ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: List models
Write-Host "`n2. Testing models endpoint..." -ForegroundColor Yellow
try {
    $modelsResponse = Invoke-RestMethod -Uri "$baseUrl/v1/models" -Method GET
    Write-Host "✓ Models endpoint passed:" -ForegroundColor Green
    $modelsResponse | ConvertTo-Json -Depth 3
} catch {
    Write-Host "✗ Models endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Chat completions
Write-Host "`n3. Testing chat completions endpoint..." -ForegroundColor Yellow

$chatRequest = @{
    model = "gpt-4-turbo"
    messages = @(
        @{
            role = "user"
            content = "What is this documentation about?"
        }
    )
    temperature = 0.7
    max_tokens = 150
} | ConvertTo-Json -Depth 3

$headers = @{
    "Content-Type" = "application/json"
}

try {
    Write-Host "Sending request to $baseUrl/v1/chat/completions..." -ForegroundColor Cyan
    Write-Host "Request body:" -ForegroundColor Gray
    Write-Host $chatRequest -ForegroundColor Gray
    
    $chatResponse = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method POST -Body $chatRequest -Headers $headers
    Write-Host "✓ Chat completions endpoint passed:" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Cyan
    $chatResponse | ConvertTo-Json -Depth 5
    
    # Extract and display just the message content
    if ($chatResponse.choices -and $chatResponse.choices.Count -gt 0) {
        Write-Host "`nAssistant Response:" -ForegroundColor Magenta
        Write-Host $chatResponse.choices[0].message.content -ForegroundColor White
    }
} catch {
    Write-Host "✗ Chat completions failed: $($_.Exception.Message)" -ForegroundColor Red
    if ($_.Exception.Response) {
        $errorStream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($errorStream)
        $errorBody = $reader.ReadToEnd()
        Write-Host "Error details: $errorBody" -ForegroundColor Red
    }
}

# Test 4: Another chat completion with a different question
Write-Host "`n4. Testing with a specific documentation question..." -ForegroundColor Yellow

$chatRequest2 = @{
    model = "gpt-4-turbo"
    messages = @(
        @{
            role = "user"
            content = "How do I make HTTP requests using this library?"
        }
    )
    temperature = 0.5
} | ConvertTo-Json -Depth 3

try {
    $chatResponse2 = Invoke-RestMethod -Uri "$baseUrl/v1/chat/completions" -Method POST -Body $chatRequest2 -Headers $headers
    Write-Host "✓ Second chat completions test passed:" -ForegroundColor Green
    
    if ($chatResponse2.choices -and $chatResponse2.choices.Count -gt 0) {
        Write-Host "`nAssistant Response:" -ForegroundColor Magenta
        Write-Host $chatResponse2.choices[0].message.content -ForegroundColor White
    }
} catch {
    Write-Host "✗ Second chat completions failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`n" + "=" * 50
Write-Host "API testing completed!" -ForegroundColor Green
