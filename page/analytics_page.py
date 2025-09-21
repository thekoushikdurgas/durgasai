"""
Analytics Page for DurgasAI Application.

Displays conversation analytics, usage metrics, and performance data.

This module provides comprehensive analytics and insights for DurgasAI usage:
- Session metrics and conversation statistics
- Message distribution analysis
- Model usage patterns and performance
- User interaction tracking
- Performance trends and optimization insights

The analytics help users understand their usage patterns and optimize
their interaction with the AI models.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from core.application_state import ApplicationState
from components.common.metrics_display import MetricsDisplay

# Import logging utilities with fallback
try:
    from utils.logger import debug, info, warning, error, log_user_action
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    # Fallback logging functions
    def debug(msg, component="analytics_page", **kwargs): pass
    def info(msg, component="analytics_page", **kwargs): pass
    def warning(msg, component="analytics_page", **kwargs): pass
    def error(msg, component="analytics_page", **kwargs): pass
    def log_user_action(action, **kwargs): pass


def render_analytics_page() -> None:
    """
    Render the analytics page with comprehensive metrics and insights.
    
    This function creates a detailed analytics dashboard showing:
    - Session metrics and conversation statistics
    - Message distribution and analysis
    - Model usage patterns and performance
    - User interaction insights
    
    All analytics are logged for meta-analysis and system optimization.
    """
    debug("Starting analytics page render", "analytics_page")
    
    try:
        # Log page access for analytics
        log_user_action("analytics_page_visited", 
                       timestamp=datetime.now().isoformat(),
                       logging_available=LOGGING_AVAILABLE)
        
        # Render main header
        debug("Rendering analytics page header", "analytics_page")
        st.markdown('<h1 class="main-header">📊 Analytics</h1>', unsafe_allow_html=True)
        
        # Get session metrics with error handling
        debug("Retrieving session metrics from ApplicationState", "analytics_page")
        try:
            metrics = ApplicationState.get_session_metrics()
            debug("Session metrics retrieved successfully", "analytics_page", 
                  metrics_count=len(metrics) if metrics else 0)
        except Exception as e:
            error("Failed to retrieve session metrics", "analytics_page", e)
            metrics = {}
            st.error("❌ Failed to load session metrics")
        
        # Display metrics using component with error handling
        debug("Rendering metrics display component", "analytics_page")
        try:
            metrics_display = MetricsDisplay()
            metrics_display.render_session_metrics(metrics)
            debug("Metrics display component rendered successfully", "analytics_page")
        except Exception as e:
            error("Failed to render metrics display component", "analytics_page", e)
            st.error("❌ Failed to display metrics")
        
        # Message distribution analysis
        debug("Rendering message analysis section", "analytics_page")
        try:
            _render_message_analysis()
            debug("Message analysis section rendered successfully", "analytics_page")
        except Exception as e:
            error("Failed to render message analysis", "analytics_page", e)
            st.error("❌ Failed to load message analysis")
        
        # Model usage statistics
        debug("Rendering model usage section", "analytics_page")
        try:
            _render_model_usage()
            debug("Model usage section rendered successfully", "analytics_page")
        except Exception as e:
            error("Failed to render model usage statistics", "analytics_page", e)
            st.error("❌ Failed to load model usage statistics")
        
        # Log successful page render
        info("Analytics page rendered successfully", "analytics_page",
             sections_rendered=["metrics", "message_analysis", "model_usage"],
             render_timestamp=datetime.now().isoformat())
        
    except Exception as e:
        # Handle any unexpected errors during page render
        error("Critical error rendering analytics page", "analytics_page", e)
        st.error("🚨 An error occurred while loading the analytics page. Please refresh and try again.")
        
        # Show basic fallback content
        st.markdown("## 📊 Analytics")
        st.markdown("Please refresh the page to access the full analytics dashboard.")


def _render_message_analysis() -> None:
    """
    Render message distribution analysis with detailed logging.
    
    This function analyzes conversation patterns including:
    - User vs assistant message counts
    - Average message lengths
    - Recent activity timeline
    - Message content analysis
    
    All analysis operations are logged for debugging and optimization.
    """
    debug("Starting message distribution analysis", "analytics_page")
    st.markdown("### 💬 Message Distribution")
    
    # Retrieve messages with error handling
    try:
        messages = ApplicationState.get("messages", [])
        debug("Retrieved messages for analysis", "analytics_page", 
              total_messages=len(messages))
    except Exception as e:
        error("Failed to retrieve messages for analysis", "analytics_page", e)
        messages = []
        st.error("❌ Failed to load conversation data")
    
    if messages:
        # Calculate message statistics with detailed logging
        debug("Calculating message statistics", "analytics_page")
        try:
            user_messages = [m for m in messages if m.get("role") == "user"]
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            
            debug("Message analysis completed", "analytics_page",
                  user_messages=len(user_messages),
                  assistant_messages=len(assistant_messages),
                  total_messages=len(messages))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("User Messages", len(user_messages))
                if user_messages:
                    # Calculate average user message length with error handling
                    try:
                        avg_length = sum(len(m.get("content", "")) for m in user_messages) / len(user_messages)
                        st.metric("Avg User Message Length", f"{avg_length:.0f} chars")
                        debug("User message statistics calculated", "analytics_page", avg_length=avg_length)
                    except Exception as e:
                        error("Failed to calculate user message statistics", "analytics_page", e)
                        st.metric("Avg User Message Length", "Error")
            
            with col2:
                st.metric("Assistant Messages", len(assistant_messages))
                if assistant_messages:
                    # Calculate average assistant message length with error handling
                    try:
                        avg_length = sum(len(m.get("content", "")) for m in assistant_messages) / len(assistant_messages)
                        st.metric("Avg Assistant Message Length", f"{avg_length:.0f} chars")
                        debug("Assistant message statistics calculated", "analytics_page", avg_length=avg_length)
                    except Exception as e:
                        error("Failed to calculate assistant message statistics", "analytics_page", e)
                        st.metric("Avg Assistant Message Length", "Error")
                        
        except Exception as e:
            error("Failed to calculate message statistics", "analytics_page", e)
            st.error("❌ Failed to analyze message distribution")
        
        # Recent activity
        st.markdown("### 🕐 Recent Activity")
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        for msg in recent_messages:
            role_icon = "👤" if msg.get("role") == "user" else "🤖"
            timestamp = msg.get("timestamp", "Unknown")
            content_preview = msg.get("content", "")[:100]
            if len(msg.get("content", "")) > 100:
                content_preview += "..."
            
            st.write(f"{role_icon} **{msg.get('role', 'unknown').title()}** ({timestamp}): {content_preview}")
    
    else:
        st.info("No conversation data available. Start chatting to see analytics!")


def _render_model_usage() -> None:
    """Render model usage statistics."""
    st.markdown("### 🤖 Model Usage")
    
    current_model = ApplicationState.get_current_model()
    model_loaded = ApplicationState.is_model_loaded()
    last_response_time = ApplicationState.get("last_model_response_time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if model_loaded and current_model:
            st.success(f"✅ **Current Model**: {current_model}")
        else:
            st.warning("⚠️ **Current Model**: None selected")
    
    with col2:
        if last_response_time:
            st.metric("Last Response Time", f"{last_response_time:.2f}s")
        else:
            st.metric("Last Response Time", "N/A")
