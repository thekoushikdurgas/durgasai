/**
 * DurgasAI JavaScript Enhancements
 * Custom JavaScript for enhanced user experience
 */

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('🤖 DurgasAI - Initializing application...');
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize UI enhancements
    initializeUIEnhancements();
    
    // Setup chat enhancements
    setupChatEnhancements();
    
    console.log('✅ DurgasAI - Application initialized successfully');
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Auto-scroll chat to bottom
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
    
    // Enhanced button interactions
    document.addEventListener('click', function(e) {
        if (e.target.matches('.stButton > button')) {
            addButtonClickEffect(e.target);
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        handleKeyboardShortcuts(e);
    });
}

/**
 * Add button click effect
 */
function addButtonClickEffect(button) {
    button.style.transform = 'scale(0.95)';
    setTimeout(() => {
        button.style.transform = '';
    }, 150);
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const chatInput = document.querySelector('input[placeholder*="message"]');
        if (chatInput) {
            const event = new KeyboardEvent('keydown', {
                key: 'Enter',
                code: 'Enter',
                keyCode: 13,
                bubbles: true
            });
            chatInput.dispatchEvent(event);
        }
    }
    
    // Ctrl/Cmd + K to focus on chat input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const chatInput = document.querySelector('input[placeholder*="message"]');
        if (chatInput) {
            chatInput.focus();
        }
    }
    
    // Escape to clear current input
    if (e.key === 'Escape') {
        const activeInput = document.activeElement;
        if (activeInput && activeInput.tagName === 'INPUT') {
            activeInput.value = '';
        }
    }
}

/**
 * Initialize UI enhancements
 */
function initializeUIEnhancements() {
    // Add loading animations
    addLoadingAnimations();
    
    // Setup smooth scrolling
    setupSmoothScrolling();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Setup theme detection
    setupThemeDetection();
}

/**
 * Add loading animations
 */
function addLoadingAnimations() {
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1) { // Element node
                    // Add fade-in animation to new elements
                    if (node.classList && !node.classList.contains('fade-in')) {
                        node.classList.add('fade-in');
                    }
                    
                    // Add slide-up animation to chat messages
                    if (node.querySelector && node.querySelector('[data-testid="chat-message"]')) {
                        node.classList.add('slide-up');
                    }
                }
            });
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

/**
 * Setup smooth scrolling
 */
function setupSmoothScrolling() {
    document.documentElement.style.scrollBehavior = 'smooth';
}

/**
 * Initialize tooltips
 */
function initializeTooltips() {
    const elements = document.querySelectorAll('[title]');
    elements.forEach(element => {
        element.addEventListener('mouseenter', showTooltip);
        element.addEventListener('mouseleave', hideTooltip);
    });
}

/**
 * Show tooltip
 */
function showTooltip(e) {
    const tooltip = document.createElement('div');
    tooltip.className = 'custom-tooltip';
    tooltip.textContent = e.target.title;
    tooltip.style.cssText = `
        position: absolute;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        z-index: 1000;
        pointer-events: none;
        white-space: nowrap;
    `;
    
    document.body.appendChild(tooltip);
    
    const rect = e.target.getBoundingClientRect();
    tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
    tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + 'px';
    
    // Remove the title to prevent default tooltip
    e.target.setAttribute('data-original-title', e.target.title);
    e.target.removeAttribute('title');
}

/**
 * Hide tooltip
 */
function hideTooltip(e) {
    const tooltip = document.querySelector('.custom-tooltip');
    if (tooltip) {
        tooltip.remove();
    }
    
    // Restore the title
    if (e.target.getAttribute('data-original-title')) {
        e.target.title = e.target.getAttribute('data-original-title');
        e.target.removeAttribute('data-original-title');
    }
}

/**
 * Setup theme detection
 */
function setupThemeDetection() {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    function handleThemeChange(e) {
        document.body.classList.toggle('dark-theme', e.matches);
    }
    
    mediaQuery.addEventListener('change', handleThemeChange);
    handleThemeChange(mediaQuery);
}

/**
 * Setup chat enhancements
 */
function setupChatEnhancements() {
    // Auto-resize text areas
    setupAutoResize();
    
    // Chat input enhancements
    setupChatInputEnhancements();
    
    // Message formatting
    setupMessageFormatting();
}

/**
 * Setup auto-resize for text areas
 */
function setupAutoResize() {
    document.addEventListener('input', function(e) {
        if (e.target.tagName === 'TEXTAREA') {
            e.target.style.height = 'auto';
            e.target.style.height = e.target.scrollHeight + 'px';
        }
    });
}

/**
 * Setup chat input enhancements
 */
function setupChatInputEnhancements() {
    // Add typing indicator
    let typingTimer;
    const typingDelay = 1000;
    
    document.addEventListener('input', function(e) {
        if (e.target.matches('input[placeholder*="message"]')) {
            clearTimeout(typingTimer);
            
            // Show typing indicator
            showTypingIndicator();
            
            // Hide typing indicator after delay
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
    const indicator = document.getElementById('typing-indicator');
    if (!indicator) {
        const newIndicator = document.createElement('div');
        newIndicator.id = 'typing-indicator';
        newIndicator.innerHTML = '💭 Typing...';
        newIndicator.style.cssText = `
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
        document.body.appendChild(newIndicator);
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
    document.addEventListener('DOMContentLoaded', function() {
        formatCodeBlocks();
    });
    
    // Format links
    formatLinks();
}

/**
 * Format code blocks
 */
function formatCodeBlocks() {
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

/**
 * Format links
 */
function formatLinks() {
    const links = document.querySelectorAll('a');
    links.forEach(link => {
        if (!link.classList.contains('formatted')) {
            link.classList.add('formatted');
            link.style.textDecoration = 'none';
            link.style.borderBottom = '1px dotted';
            link.style.transition = 'all 0.3s ease';
            
            link.addEventListener('mouseenter', function() {
                this.style.borderBottom = '1px solid';
            });
            
            link.addEventListener('mouseleave', function() {
                this.style.borderBottom = '1px dotted';
            });
        }
    });
}

/**
 * Utility functions
 */

/**
 * Show notification
 */
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `toast ${type}`;
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
            <span>${getNotificationIcon(type)}</span>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, duration);
}

/**
 * Get notification icon
 */
function getNotificationIcon(type) {
    const icons = {
        success: '✅',
        error: '❌',
        warning: '⚠️',
        info: 'ℹ️'
    };
    return icons[type] || icons.info;
}

/**
 * Copy text to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy to clipboard', 'error');
    });
}

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

/**
 * Debounce function
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Add CSS animations
 */
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    .slide-up {
        animation: slideUp 0.3s ease-out;
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
`;
document.head.appendChild(style);

// Export functions for global use
window.DurgasAI = {
    showNotification,
    copyToClipboard,
    formatTimestamp,
    debounce,
    throttle
};
