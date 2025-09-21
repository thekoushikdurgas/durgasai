"""
Application State Management for DurgasAI.

This module provides centralized state management for the DurgasAI application.
It acts as a wrapper around Streamlit's session state with additional features:

Key Features:
- Centralized state initialization with default values
- Type-safe state access methods
- Logging integration for state changes
- Session metrics and analytics
- Batch state updates
- State validation and cleanup

Architecture:
- Uses Streamlit's session_state as the underlying storage
- Provides a clean API for state operations
- Integrates with the logging system for debugging
- Supports both individual and batch state operations

Usage:
- ApplicationState.initialize() - Initialize state with defaults
- ApplicationState.get(key, default) - Get state value
- ApplicationState.set(key, value) - Set state value
- ApplicationState.update(dict) - Batch update state
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Add utils to path for proper module resolution
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, log_session_event


class ApplicationState:
    """
    Centralized application state management.
    
    This class provides a clean interface for managing application state
    while keeping UI concerns separate from state logic.
    """
    
    @staticmethod
    def initialize() -> None:
        """
        Initialize application state with default values.
        
        This method sets up the initial state for the application with sensible
        defaults. It only initializes keys that don't already exist, allowing
        for state persistence across reruns.
        
        State Categories:
        - Core application state: Basic app status and configuration
        - Session management: Session tracking and identification
        - Model state: AI model status and configuration
        - Chat state: Conversation history and management
        - Performance tracking: Metrics and timing data
        
        Note:
            This method is idempotent - it can be called multiple times safely
            and will only initialize missing keys.
        """
        debug("Starting application state initialization", "core")
        
        # Define default state values organized by category
        defaults = {
            # Core application state
            # These control the basic application behavior and status
            "app_initialized": False,
            "current_page": "home",
            "debug_mode": False,
            
            # Session management
            # These track user sessions for analytics and debugging
            "session_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:17],
            "session_start_time": datetime.now(),
            
            # Model state
            # These track the current AI model status and configuration
            "model_loaded": False,
            "current_model": None,
            "api_token": "",
            "system_prompt": "",
            
            # Chat state
            # These manage conversation history and chat sessions
            "messages": [],
            "total_messages": 0,
            "chat_session_id": "default",
            
            # Performance tracking
            # These store performance metrics for optimization
            "last_model_response_time": None,
            "last_interaction_time": None,
        }
        
        debug(f"Defined {len(defaults)} default state keys", "core")
        
        # Initialize only missing keys to preserve existing state
        new_keys = []
        existing_keys = []
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                new_keys.append(key)
                debug(f"Initialized new state key: {key}", "core")
            else:
                existing_keys.append(key)
        
        debug(f"State initialization complete: {len(new_keys)} new, {len(existing_keys)} existing", "core")
        
        # Log initialization results
        if new_keys:
            info(f"Application state initialized: {len(new_keys)} new keys added", "core")
            debug(f"New keys: {', '.join(new_keys)}", "core")
            log_session_event("app_state_initialized", new_keys_count=len(new_keys))
        else:
            debug("No new state keys needed - using existing state", "core")
        
        # Mark application as initialized
        st.session_state.app_initialized = True
        debug("Application state marked as initialized", "core")
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a value from application state.
        
        Args:
            key: The state key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The state value or default if key not found
            
        Note:
            This method provides safe access to state values with fallback defaults.
        """
        value = st.session_state.get(key, default)
        debug(f"Retrieved state value for key '{key}': {type(value).__name__}", "core")
        return value
    
    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in application state.
        
        Args:
            key: The state key to set
            value: The value to store
            
        Note:
            This method logs state changes for debugging purposes.
        """
        old_value = st.session_state.get(key, "<not_set>")
        st.session_state[key] = value
        debug(f"State key '{key}' updated: {old_value} -> {value}", "core")
    
    @staticmethod
    def update(updates: Dict[str, Any]) -> None:
        """
        Update multiple values in application state.
        
        Args:
            updates: Dictionary of key-value pairs to update
            
        Note:
            This method is more efficient than multiple individual set() calls
            and provides batch logging for debugging.
        """
        debug(f"Starting batch state update for {len(updates)} keys", "core")
        
        for key, value in updates.items():
            old_value = st.session_state.get(key, "<not_set>")
            st.session_state[key] = value
            debug(f"Batch update - {key}: {old_value} -> {value}", "core")
        
        info(f"Completed batch state update for keys: {list(updates.keys())}", "core")
    
    @staticmethod
    def clear() -> None:
        """Clear all application state."""
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            del st.session_state[key]
        info(f"Application state cleared: {len(keys_to_clear)} keys", "core")
    
    @staticmethod
    def is_model_loaded() -> bool:
        """Check if a model is currently loaded."""
        return st.session_state.get("model_loaded", False)
    
    @staticmethod
    def get_current_model() -> Optional[str]:
        """Get the name of the currently loaded model."""
        return st.session_state.get("current_model")
    
    @staticmethod
    def get_session_metrics() -> Dict[str, Any]:
        """Get session metrics for analytics."""
        session_start = st.session_state.get("session_start_time", datetime.now())
        duration = datetime.now() - session_start
        
        return {
            "session_id": st.session_state.get("session_id"),
            "session_duration": str(duration).split('.')[0],
            "total_messages": st.session_state.get("total_messages", 0),
            "current_messages": len(st.session_state.get("messages", [])),
            "current_model": st.session_state.get("current_model", "None"),
            "model_loaded": st.session_state.get("model_loaded", False),
            "last_response_time": st.session_state.get("last_model_response_time"),
        }
