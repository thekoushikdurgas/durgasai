/**
 * Main JavaScript for DurgasAI Application
 * Consolidated and organized JavaScript functionality.
 */

// Import modular JS components
import './chat.js';
import './utils.js';

// Global DurgasAI namespace
window.DurgasAI = window.DurgasAI || {};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🤖 DurgasAI - Initializing application');
    initializeApp();
});

/**
 * Initialize the main application
 */
function initializeApp() {
    try {
        // Setup core functionality
        setupEventListeners();
        setupUIEnhancements();
        setupPerformanceMonitoring();
        
        console.log('✅ DurgasAI JavaScript initialized successfully');
        
    } catch (error) {
        console.error('❌ Error initializing DurgasAI JavaScript:', error);
    }
}

/**
 * Setup global event listeners
 */
function setupEventListeners() {
    // Enhanced button interactions
    document.addEventListener('click', function(e) {
        if (e.target.matches('.stButton > button')) {
            addButtonClickEffect(e.target);
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

/**
 * Setup UI enhancements
 */
function setupUIEnhancements() {
    // Add loading animations
    addLoadingAnimations();
    
    // Setup smooth scrolling
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Initialize tooltips
    initializeTooltips();
}

/**
 * Handle keyboard shortcuts
 */
function handleKeyboardShortcuts(e) {
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
 * Add button click effect
 */
function addButtonClickEffect(button) {
    button.style.transform = 'scale(0.95)';
    setTimeout(() => {
        button.style.transform = '';
    }, 150);
}

/**
 * Add loading animations
 */
function addLoadingAnimations() {
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === 1 && node.classList) {
                    node.classList.add('fade-in');
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
 * Initialize tooltips
 */
function initializeTooltips() {
    // Custom tooltip implementation
    document.addEventListener('mouseenter', function(e) {
        if (e.target.hasAttribute('title')) {
            showCustomTooltip(e);
        }
    }, true);
    
    document.addEventListener('mouseleave', hideCustomTooltip, true);
}

/**
 * Setup performance monitoring
 */
function setupPerformanceMonitoring() {
    if (window.performance) {
        // Monitor page load time
        const loadTime = performance.timing.loadEventEnd - performance.timing.navigationStart;
        console.log(`Page load time: ${loadTime}ms`);
        
        // Monitor memory usage periodically
        if (performance.memory) {
            setInterval(() => {
                const memUsage = (performance.memory.usedJSHeapSize / 1024 / 1024).toFixed(2);
                console.log(`Memory usage: ${memUsage} MB`);
            }, 30000);
        }
    }
}

// Export main functions
window.DurgasAI.main = {
    initializeApp,
    setupEventListeners,
    setupUIEnhancements
};
