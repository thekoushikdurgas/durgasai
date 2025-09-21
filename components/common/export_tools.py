"""
Export Tools Component.

Handles data export functionality for conversations and settings.
"""

import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation


class ExportTools:
    """Component for handling data export operations."""
    
    def __init__(self):
        """Initialize export tools."""
        debug("ExportTools initialized", "export_tools")
    
    def render_export_button(self, messages: List[Dict[str, Any]], filename: str = None) -> None:
        """
        Render export button for chat history with comprehensive logging and error handling.
        
        This method creates a download button for exporting chat history to a JSON file.
        It includes validation, error handling, and detailed logging for debugging.
        
        Args:
            messages: List of chat messages to export
            filename: Optional custom filename for the export
            
        Features:
        - Chat history validation and processing
        - JSON export with proper formatting
        - Error handling for malformed data
        - User interaction tracking
        - File naming with timestamps
        
        Logging:
        - Export button rendering status
        - Message validation and processing
        - Export operation details
        - Error handling for export failures
        - User download analytics
        """
        debug("Starting chat history export button rendering", "export_tools",
              message_count=len(messages) if messages else 0,
              custom_filename=filename is not None)
        
        try:
            with LoggedOperation("export_button_rendering", "export_tools"):
                # Validate input data
                if not messages:
                    debug("No messages to export, showing info message", "export_tools")
                    st.info("No messages to export")
                    log_user_action("export_attempted_no_messages")
                    return
                
                # Validate message structure
                valid_messages = []
                invalid_count = 0
                
                for i, message in enumerate(messages):
                    if isinstance(message, dict) and 'content' in message and 'role' in message:
                        valid_messages.append(message)
                    else:
                        invalid_count += 1
                        warning(f"Invalid message at index {i}", "export_tools",
                               message_type=type(message).__name__,
                               message_keys=list(message.keys()) if isinstance(message, dict) else "not_dict")
                
                debug("Message validation completed", "export_tools",
                      total_messages=len(messages),
                      valid_messages=len(valid_messages),
                      invalid_messages=invalid_count)
                
                if not valid_messages:
                    warning("No valid messages found for export", "export_tools")
                    st.warning("No valid messages found to export")
                    log_user_action("export_failed_no_valid_messages", invalid_count=invalid_count)
                    return
                
                # Generate filename with timestamp
                if filename is None:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"chat_history_{timestamp}.json"
                    debug("Generated export filename", "export_tools", filename=filename)
                
                # Prepare export data with metadata
                chat_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "export_version": "1.0",
                    "total_messages": len(valid_messages),
                    "invalid_messages_skipped": invalid_count,
                    "export_source": "durgasai_chat_interface",
                    "messages": valid_messages
                }
                
                debug("Export data prepared", "export_tools",
                      export_timestamp=chat_data["export_timestamp"],
                      total_messages=chat_data["total_messages"],
                      invalid_skipped=chat_data["invalid_messages_skipped"])
                
                # Convert to JSON with error handling
                try:
                    json_data = json.dumps(chat_data, indent=2, ensure_ascii=False)
                    json_size = len(json_data.encode('utf-8'))
                    debug("JSON data generated successfully", "export_tools",
                          json_size_bytes=json_size,
                          json_size_mb=round(json_size / (1024 * 1024), 2))
                except Exception as e:
                    error("Error generating JSON export data", "export_tools", e,
                          message_count=len(valid_messages))
                    st.error("🚨 Error preparing export data. Please try again.")
                    log_user_action("export_failed_json_generation", error=str(e))
                    return
                
                # Render download button
                try:
                    if st.download_button(
                        label="📥 Download Chat History",
                        data=json_data,
                        file_name=filename,
                        mime="application/json",
                        help=f"Download {len(valid_messages)} messages as a JSON file",
                        use_container_width=True,
                        key="chat_history_export"
                    ):
                        debug("Chat history export initiated", "export_tools",
                              filename=filename,
                              message_count=len(valid_messages),
                              file_size_mb=round(json_size / (1024 * 1024), 2))
                        
                        # Log successful export action
                        log_user_action("chat_history_exported",
                                      message_count=len(valid_messages),
                                      filename=filename,
                                      file_size_bytes=json_size,
                                      invalid_messages_skipped=invalid_count)
                        
                        debug("Chat history export completed successfully", "export_tools")
                    
                    debug("Export download button rendered successfully", "export_tools")
                    
                except Exception as e:
                    error("Error rendering download button", "export_tools", e,
                          filename=filename,
                          message_count=len(valid_messages))
                    st.error("🚨 Error creating download button. Please try again.")
                    log_user_action("export_failed_button_rendering", error=str(e))
                    return
                
        except Exception as e:
            error("Critical error in export button rendering", "export_tools", e,
                  message_count=len(messages) if messages else 0,
                  filename=filename)
            st.error("🚨 An error occurred while preparing the export. Please try again.")
            log_user_action("export_failed_critical_error", error=str(e))
            raise
        
        debug("Chat history export button rendering completed", "export_tools")
    
    def render_settings_export(self, settings: Dict[str, Any]) -> None:
        """Render export button for application settings."""
        settings_data = {
            "export_timestamp": datetime.now().isoformat(),
            "settings": settings
        }
        
        json_data = json.dumps(settings_data, indent=2, ensure_ascii=False)
        filename = f"durgasai_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        st.download_button(
            label="📥 Export Settings",
            data=json_data,
            file_name=filename,
            mime="application/json",
            help="Export current application settings",
            use_container_width=True
        )
