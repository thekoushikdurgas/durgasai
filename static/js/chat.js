/**
 * Chat-specific JavaScript functionality
 * Handles chat interface enhancements and interactions.
 */

/**
 * Setup chat-specific enhancements
 */
function setupChatEnhancements() {
    setupAutoScroll();
    setupTypingIndicator();
    setupMessageFormatting();
}

/**
 * Setup auto-scroll for chat container
 */
function setupAutoScroll() {
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        const observer = new MutationObserver(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        });
        
        observer.observe(chatContainer, {
            childList: true,
            subtree: true
        });
    }
}

/**
 * Setup typing indicator
 */
function setupTypingIndicator() {
    let typingTimer;
    const typingDelay = 1000;
    
    document.addEventListener('input', function(e) {
        if (e.target.matches('input[placeholder*="message"]')) {
            clearTimeout(typingTimer);
            showTypingIndicator();
            
            typingTimer = setTimeout(() => {
                hideTypingIndicator();
            }, typingDelay);
        }
    });
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    if (!document.getElementById('typing-indicator')) {
        const indicator = document.createElement('div');
        indicator.id = 'typing-indicator';
        indicator.innerHTML = '💭 Typing...';
        indicator.style.cssText = `
            position: fixed;
            bottom: 100px;
            right: 20px;
            background: rgba(31, 119, 180, 0.9);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            z-index: 1000;
            animation: fadeIn 0.3s ease-in;
        `;
        document.body.appendChild(indicator);
    }
}

/**
 * Hide typing indicator
 */
function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => indicator.remove(), 300);
    }
}

/**
 * Setup message formatting
 */
function setupMessageFormatting() {
    // Format code blocks
    const codeBlocks = document.querySelectorAll('code');
    codeBlocks.forEach(block => {
        if (!block.classList.contains('formatted')) {
            block.classList.add('formatted');
            block.style.cssText = `
                background: rgba(0, 0, 0, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                font-size: 0.9em;
            `;
        }
    });
}

// Export chat functions
window.DurgasAI.chat = {
    setupChatEnhancements,
    showTypingIndicator,
    hideTypingIndicator
};
