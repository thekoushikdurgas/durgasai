"""
Metrics Display Component.

Handles the display of various metrics and model information.
"""

import streamlit as st
from typing import Dict, Any
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.config import Config
from utils.logger import debug, info, warning, error, log_user_action, LoggedOperation


class MetricsDisplay:
    """Component for displaying metrics and statistics."""
    
    def __init__(self):
        """Initialize metrics display component."""
        debug("MetricsDisplay initialized", "metrics_display")
    
    def render_session_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Render session metrics in a dashboard format with comprehensive logging.
        
        This method displays session statistics in a structured dashboard layout
        with proper error handling and detailed logging for debugging.
        
        Args:
            metrics: Dictionary of metrics to display
            
        Features:
        - Dashboard-style metrics display
        - Data validation and error handling
        - Performance logging for large datasets
        - User interaction tracking
        - Responsive column layout
        
        Logging:
        - Metrics rendering status
        - Data validation results
        - Display performance
        - Error handling for malformed data
        - User interaction analytics
        """
        debug("Starting session metrics rendering", "metrics_display",
              metrics_count=len(metrics) if metrics else 0)
        
        try:
            with LoggedOperation("session_metrics_rendering", "metrics_display"):
                # Validate input data
                if not metrics:
                    warning("No metrics provided for display", "metrics_display")
                    st.info("No session metrics available")
                    log_user_action("metrics_displayed_empty")
                    return
                
                # Validate metrics data structure
                valid_metrics = {}
                invalid_count = 0
                
                for key, value in metrics.items():
                    if key and value is not None:
                        valid_metrics[key] = value
                    else:
                        invalid_count += 1
                        warning(f"Invalid metric entry: {key}={value}", "metrics_display")
                
                debug("Metrics validation completed", "metrics_display",
                      total_metrics=len(metrics),
                      valid_metrics=len(valid_metrics),
                      invalid_metrics=invalid_count)
                
                if not valid_metrics:
                    warning("No valid metrics found for display", "metrics_display")
                    st.warning("No valid session metrics available")
                    log_user_action("metrics_displayed_no_valid_data", invalid_count=invalid_count)
                    return
                
                # Render metrics header
                st.markdown("### 📊 Session Statistics")
                debug("Metrics header rendered", "metrics_display")
                
                # Create responsive column layout
                metric_count = len(valid_metrics)
                cols = st.columns(metric_count)
                debug("Metrics columns created", "metrics_display", column_count=metric_count)
                
                # Display metrics with error handling
                displayed_count = 0
                for i, (key, value) in enumerate(valid_metrics.items()):
                    try:
                        with cols[i]:
                            # Format label from key
                            label = key.replace('_', ' ').title()
                            
                            # Format value for display
                            if isinstance(value, (int, float)):
                                if isinstance(value, float):
                                    display_value = f"{value:.2f}" if value != int(value) else str(int(value))
                                else:
                                    display_value = str(value)
                            else:
                                display_value = str(value)
                            
                            debug(f"Rendering metric {i+1}/{metric_count}", "metrics_display",
                                  key=key,
                                  label=label,
                                  value=display_value,
                                  value_type=type(value).__name__)
                            
                            st.metric(
                                label=label,
                                value=display_value
                            )
                            
                            displayed_count += 1
                            debug(f"Metric {i+1} displayed successfully", "metrics_display")
                            
                    except Exception as e:
                        error(f"Error displaying metric {key}", "metrics_display", e,
                              key=key,
                              value=value,
                              value_type=type(value).__name__)
                        # Continue with other metrics
                        continue
                
                debug("Session metrics rendering completed", "metrics_display",
                      total_metrics=metric_count,
                      displayed_metrics=displayed_count,
                      failed_metrics=metric_count - displayed_count)
                
                # Log successful metrics display
                log_user_action("session_metrics_displayed",
                              metric_count=displayed_count,
                              total_available=len(metrics),
                              invalid_skipped=invalid_count)
                
        except Exception as e:
            error("Critical error in session metrics rendering", "metrics_display", e,
                  metrics_count=len(metrics) if metrics else 0)
            st.error("🚨 Error displaying session metrics")
            log_user_action("metrics_display_failed", error=str(e))
            raise
        
        debug("Session metrics rendering process completed", "metrics_display")
    
    def render_model_metrics(self, model_name: str, response_time: float = None) -> None:
        """Render model-specific metrics."""
        st.markdown("### 🤖 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Model", model_name or "None")
        
        with col2:
            if response_time:
                st.metric("Last Response Time", f"{response_time:.2f}s")
            else:
                st.metric("Last Response Time", "N/A")


def display_model_overview() -> None:
    """Display overview of available models."""
    cols = st.columns(2)
    models = Config.AVAILABLE_MODELS
    
    for i, (key, model_config) in enumerate(models.items()):
        with cols[i % 2]:
            st.markdown(f"""
            **{model_config.name}**
            - **Provider**: {model_config.provider.value}
            - **Description**: {model_config.description}
            - **Max Tokens**: {model_config.max_tokens}
            """)
            
            if i < len(models) - 1:
                st.markdown("---")
