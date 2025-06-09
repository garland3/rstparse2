const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const rerankCheckbox = document.getElementById('rerankCheckbox');
const loading = document.getElementById('loading');

function addMessage(content, isUser = false, isError = false, sources = []) {
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
        
        // Add sources if available
        if (sources && sources.length > 0) {
            const sourcesContainer = createSourcesContainer(sources);
            contentDiv.appendChild(sourcesContainer);
        }
    }
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function createSourcesContainer(sources) {
    const sourcesContainer = document.createElement('div');
    sourcesContainer.className = 'sources-container';
    
    // Create header
    const sourcesHeader = document.createElement('div');
    sourcesHeader.className = 'sources-header';
    sourcesHeader.innerHTML = `
        <span class="sources-toggle">▶</span>
        <span>View Sources</span>
        <span class="sources-count">${sources.length}</span>
    `;
    
    // Create content container
    const sourcesContent = document.createElement('div');
    sourcesContent.className = 'sources-content';
    
    // Add each source chunk
    sources.forEach((source, index) => {
        const sourceChunk = document.createElement('div');
        sourceChunk.className = 'source-chunk';
        
        const sourceHeader = document.createElement('div');
        sourceHeader.className = 'source-chunk-header';
        sourceHeader.textContent = `Source ${index + 1}`;
        
        const sourceContent = document.createElement('div');
        sourceContent.className = 'source-chunk-content';
        sourceContent.textContent = source;
        
        sourceChunk.appendChild(sourceHeader);
        sourceChunk.appendChild(sourceContent);
        sourcesContent.appendChild(sourceChunk);
    });
    
    // Add click handler to toggle sources
    sourcesHeader.addEventListener('click', () => {
        const toggle = sourcesHeader.querySelector('.sources-toggle');
        const isExpanded = sourcesContent.classList.contains('expanded');
        
        if (isExpanded) {
            sourcesContent.classList.remove('expanded');
            toggle.classList.remove('expanded');
            toggle.textContent = '▶';
        } else {
            sourcesContent.classList.add('expanded');
            toggle.classList.add('expanded');
            toggle.textContent = '▼';
        }
    });
    
    sourcesContainer.appendChild(sourcesHeader);
    sourcesContainer.appendChild(sourcesContent);
    
    return sourcesContainer;
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
            addMessage(data.response || 'No response received.', false, false, data.sources || []);
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
