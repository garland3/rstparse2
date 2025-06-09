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
