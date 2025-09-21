"""
DurgasAI - Advanced AI Agent Application
Main Streamlit application with navigation and model management.

This is the main entry point for the DurgasAI application. It handles:
- Application initialization and setup
- Page routing and navigation
- Session state management
- Model manager coordination
- UI component rendering

Architecture:
- Uses Streamlit for web interface
- Integrates HuggingFace models via LangChain
- Supports multiple AI models and providers
- Provides tool execution and workflow capabilities
"""

# Warning suppression is handled in utils/config.py automatically
# No need to import suppress_warnings here as it causes circular import issues

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import traceback

# Add utils to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import core utilities and configuration
from utils.config import Config
from utils.ui_helpers import UIHelpers, SessionManager
from utils.logger import info, debug, error, warning, log_session_event, log_user_action, LoggedOperation

# Import new service architecture
from services import ServiceManager

# Import Auto Classes integration (with fallback)
try:
    from utils.auto_classes_integration import DurgasAIAutoClassesIntegrator
    AUTO_CLASSES_AVAILABLE = True
    info("Auto Classes integration available", "app")
except ImportError as e:
    AUTO_CLASSES_AVAILABLE = False
    debug(f"Auto Classes integration not available: {e}", "app")

# Import logging with fallback
# try:
# except ImportError as e:
#     print(f"Warning: Could not import enhanced logging: {e}")
#     # Fallback logging functions
#     def info(msg, component="app", **kwargs): print(f"INFO: {msg}")
#     def debug(msg, component="app", **kwargs): print(f"DEBUG: {msg}")
#     def error(msg, component="app", error_obj=None, **kwargs): print(f"ERROR: {msg}")
#     def warning(msg, component="app", **kwargs): print(f"WARNING: {msg}")
#     def log_session_event(event, **kwargs): pass
#     def log_user_action(action, **kwargs): pass
    
#     class LoggedOperation:
#         def __init__(self, *args, **kwargs): pass
#         def __enter__(self): return self
#         def __exit__(self, *args): pass

# Import page components
from page.aiagent import AIAgentPage

# Import facial AI page with fallback
try:
    from page.facial_ai_page import render_facial_ai_page
except ImportError:
    def render_facial_ai_page():
        st.error("Facial AI page not available - missing dependencies")

# Import HuggingFace page
try:
    from page.huggingface_page import render_huggingface_page
except ImportError:
    def render_huggingface_page():
        st.error("HuggingFace page not available - missing dependencies")

# Import Custom Model page
try:
    from page.custom_model_page import render_custom_model_page
except ImportError:
    def render_custom_model_page():
        st.error("Custom Model page not available - missing dependencies")

# Import Component Customization page
try:
    from page.component_customization_page import render_component_customization_page
except ImportError:
    def render_component_customization_page():
        st.error("Component Customization page not available - missing dependencies")

# Import debug dashboard with fallback
try:
    from page.debug_dashboard import render_debug_dashboard
except ImportError:
    def render_debug_dashboard():
        st.error("Debug dashboard not available - missing dependencies")

# Import Auto Classes page with fallback
try:
    from page.auto_classes_page import render_auto_classes_page
except ImportError:
    def render_auto_classes_page():
        st.error("Auto Classes page not available - missing dependencies")

# Import Web Server Inference page with fallback
try:
    from page.web_server_inference_page import render_web_server_inference_page
except ImportError:
    def render_web_server_inference_page():
        st.error("Web Server Inference page not available - missing dependencies")

# Import additional pages with fallbacks
try:
    from page.chat_page import render_chat_page
except ImportError:
    def render_chat_page(model_manager):
        st.error("Chat page not available - missing dependencies")

try:
    from page.tokenizers_page import render_tokenizers_page
except ImportError:
    def render_tokenizers_page():
        st.error("Tokenizers page not available - missing dependencies")

try:
    from page.tokenizer_summary_page import render_tokenizer_summary_page
except ImportError:
    def render_tokenizer_summary_page():
        st.error("Tokenizer Summary page not available - missing dependencies")

try:
    from page.padding_truncation_page import render_padding_truncation_page
except ImportError:
    def render_padding_truncation_page():
        st.error("Padding & Truncation page not available - missing dependencies")

try:
    from page.image_processors_page import render_image_processors_page
except ImportError:
    def render_image_processors_page():
        st.error("Image Processors page not available - missing dependencies")

try:
    from page.video_processors_page import render_video_processors_page
except ImportError:
    def render_video_processors_page():
        st.error("Video Processors page not available - missing dependencies")

try:
    from page.backbones_page import render_backbones_page
except ImportError:
    def render_backbones_page():
        st.error("Backbones page not available - missing dependencies")

try:
    from page.processors_page import render_processors_page
except ImportError:
    def render_processors_page():
        st.error("Processors page not available - missing dependencies")

try:
    from page.feature_extractors_page import render_feature_extractors_page
except ImportError:
    def render_feature_extractors_page():
        st.error("Feature Extractors page not available - missing dependencies")

try:
    from page.pipeline_page import render_pipeline_page
except ImportError:
    def render_pipeline_page():
        st.error("Pipeline page not available - missing dependencies")

try:
    from page.ml_apps_page import render_ml_apps_page
except ImportError:
    def render_ml_apps_page():
        st.error("ML Apps page not available - missing dependencies")

try:
    from page.preprocessor_dashboard import render_preprocessor_dashboard
except ImportError:
    def render_preprocessor_dashboard():
        st.error("Preprocessor Dashboard not available - missing dependencies")

try:
    from page.analytics_page import render_analytics_page
except ImportError:
    def render_analytics_page():
        st.error("Analytics page not available - missing dependencies")

try:
    from page.debug_page import render_debug_page
except ImportError:
    def render_debug_page():
        st.error("Debug page not available - missing dependencies")

try:
    from page.settings_page import render_settings_page
except ImportError:
    def render_settings_page():
        st.error("Settings page not available - missing dependencies")

try:
    from page.help_page import render_help_page
except ImportError:
    def render_help_page():
        st.error("Help page not available - missing dependencies")

try:
    from page.home_page import render_home_page
except ImportError:
    def render_home_page():
        st.error("Home page not available - missing dependencies")


class DurgasAIApp:
    """
    Main application class that orchestrates the entire DurgasAI system.
    
    This class is responsible for:
    - Initializing the service manager and all application services
    - Setting up the Streamlit application configuration
    - Managing page routing and navigation
    - Coordinating between different modules and services
    
    Attributes:
        service_manager (ServiceManager): Central service coordinator
        model_service (ModelService): AI model operations and management
        chat_service (ChatService): Chat functionality and conversation management
        analytics_service (AnalyticsService): Metrics and usage tracking
        config_service (ConfigService): Configuration management
        model_manager (ModelManager): Legacy compatibility reference
    """
    
    def __init__(self):
        """
        Initialize the DurgasAI application with new service architecture.
    
    This constructor performs the complete application initialization sequence
    using the new service-oriented architecture for better separation of concerns.
    
    Initialization Sequence:
    1. Load and validate HuggingFace configuration
    2. Initialize service manager and all application services
    3. Setup service references for backward compatibility
    4. Setup Auto Classes integration (if available)
    5. Log initialization events for analytics
    
    Services Initialized:
    - ConfigService: Centralized configuration management
    - AnalyticsService: Metrics and usage tracking
    - ModelService: AI model operations and management
    - ChatService: Chat functionality and conversation management
    - DurgasAIAutoClassesIntegrator: Enhanced model support (optional)
    
    The service architecture provides:
    - Clean separation of concerns
    - Centralized configuration management
    - Comprehensive analytics and monitoring
    - Async support for better performance
    - Proper dependency injection
    
    Raises:
        Exception: If critical initialization steps fail
        """
        info("Starting DurgasAI application initialization sequence", "app")
        debug("Application initialization beginning", "app")
        
        try:
            with LoggedOperation("app_initialization", "app"):
                # Step 1: Load HuggingFace configuration and setup cache directories
                debug("Step 1/4: Loading HuggingFace configuration", "app")
                hf_config = Config.load_huggingface_config()
                debug(f"HuggingFace config loaded with cache_dir: {hf_config.get('cache_dir', 'default')}", "app")
                info("HuggingFace cache configuration loaded successfully", "app")
                
                # Step 2: Initialize the service manager and all services
                debug("Step 2/4: Initializing service manager", "app")
                self.service_manager = ServiceManager()
                debug("Service manager instance created", "app")
                
                # Initialize services asynchronously
                import asyncio
                services_initialized = asyncio.run(self.service_manager.initialize_services())
                
                if not services_initialized:
                    raise Exception("Failed to initialize application services")
                
                debug("All services initialized successfully", "app")
                info("Service manager and all services initialized successfully", "app")
                
                # Step 3: Get service references for backward compatibility
                debug("Step 3/4: Setting up service references", "app")
                self.model_service = self.service_manager.get_model_service()
                self.chat_service = self.service_manager.get_chat_service()
                self.analytics_service = self.service_manager.get_analytics_service()
                self.config_service = self.service_manager.get_config_service()
                
                # Legacy model_manager reference for backward compatibility
                # TODO: Remove this once all code is refactored to use services
                self.model_manager = self.model_service.model_manager if self.model_service else None
                
                debug("Service references established", "app")
                
                # Step 4: Initialize Auto Classes integration if available
                debug("Step 4/4: Setting up Auto Classes integration", "app")
                if AUTO_CLASSES_AVAILABLE:
                    debug("Auto Classes integration is available, attempting initialization", "app")
                    try:
                        self.auto_classes_integrator = DurgasAIAutoClassesIntegrator(
                            "config/model_config.json"
                        )
                        debug("Auto Classes integrator created with enhanced config", "app")
                        info("Auto Classes integration initialized successfully", "app")
                        log_session_event("auto_classes_initialized", component="DurgasAIApp")
                    except Exception as e:
                        warning(f"Auto Classes integration failed: {str(e)}", "app")
                        debug(f"Auto Classes initialization error details: {traceback.format_exc()}", "app")
                        self.auto_classes_integrator = None
                else:
                    self.auto_classes_integrator = None
                    debug("Auto Classes integration not available - using standard models only", "app")
                
                # Log successful initialization
                log_session_event("app_initialized", component="DurgasAIApp")
                info("DurgasAI application initialization completed successfully", "app")
            
        except Exception as e:
            error("Critical failure during DurgasAI application initialization", "app", e)
            debug(f"Initialization error details: {traceback.format_exc()}", "app")
            raise
        
    def setup_application(self):
        """
        Setup the Streamlit application with all necessary configurations.
        
        This method performs the following setup steps:
        1. Configure Streamlit page settings (title, icon, layout)
        2. Load custom CSS styling for enhanced UI
        3. Initialize session state with default values
        
        Each step is logged for debugging purposes.
        """
        debug("Starting application setup", "app")
        
        with LoggedOperation("application_setup", "app"):
            try:
                # Step 1: Configure Streamlit page settings
                debug("Setting up page configuration", "app")
                UIHelpers.setup_page_config()
                info("Page configuration completed", "app")
                
                # Step 2: Load custom CSS for enhanced styling
                debug("Loading custom CSS styles", "app")
                UIHelpers.load_custom_css()
                info("Custom CSS loaded successfully", "app")
                
                # Step 3: Initialize session state with default values
                debug("Initializing session state", "app")
                SessionManager.initialize_session_state()
                info("Session state initialized", "app")
                
                # Log current session state keys for debugging
                session_keys = list(st.session_state.keys()) if hasattr(st, 'session_state') else []
                debug(f"Session state keys initialized: {session_keys}", "app")
                
                log_session_event("setup_completed", 
                    session_keys_count=len(session_keys),
                    session_keys=session_keys)
                
            except Exception as e:
                error("Application setup failed", "app", e)
                raise
    
    
    def run(self):
        """
        Run the main DurgasAI application.
        
        This is the main application loop that:
        1. Sets up the application environment
        2. Handles page navigation and routing
        3. Renders the appropriate page based on user selection
        4. Manages the overall application lifecycle
        
        The method includes comprehensive logging for debugging and monitoring.
        """
        info("Starting DurgasAI application", "app")
        
        try:
            # Setup the application environment
            with LoggedOperation("app_setup", "app"):
                self.setup_application()
            
            # Get current page selection from sidebar navigation
            debug("Creating sidebar navigation", "app")
            selected_page = UIHelpers.create_sidebar_navigation()
            
            # Log page navigation for analytics
            log_user_action("page_navigation",
                selected_page=selected_page,
                previous_page=st.session_state.get("current_page", "unknown"))
            
            # Update current page in session state for tracking
            st.session_state.current_page = selected_page
            debug(f"Navigating to page: {selected_page}", "app")
            
            # Route to appropriate page based on selection
            # Each page is wrapped with error handling and performance monitoring
            try:
                if selected_page == "home":
                    info("Rendering home page", "app")
                    with LoggedOperation("render_home_page", "app"):
                        render_home_page()
                        
                elif selected_page == "ai_agent":
                    info("Rendering AI agent page", "app")
                    with LoggedOperation("render_ai_agent_page", "app", extra_data={"model_loaded": st.session_state.get("model_loaded", False)}):
                        # Initialize AI agent page with model manager
                        ai_agent_page = AIAgentPage(self.service_manager)
                        ai_agent_page.render()
                        
                elif selected_page == "huggingface":
                    info("Rendering HuggingFace page", "app")
                    with LoggedOperation("render_huggingface_page", "app"):
                        render_huggingface_page()
                        
                elif selected_page == "model_catalog":
                    info("Rendering model catalog page", "app")
                    with LoggedOperation("render_model_catalog_page", "app"):
                        from page.model_catalog_page import render_model_catalog_page
                        render_model_catalog_page()
                        
                elif selected_page == "facial_ai":
                    info("Rendering facial AI page", "app")
                    with LoggedOperation("render_facial_ai_page", "app"):
                        render_facial_ai_page()
                        
                elif selected_page == "custom_models":
                    info("Rendering custom models page", "app")
                    with LoggedOperation("render_custom_models_page", "app"):
                        render_custom_model_page()
                        
                elif selected_page == "component_customization":
                    info("Rendering component customization page", "app")
                    with LoggedOperation("render_component_customization_page", "app"):
                        render_component_customization_page()
                        
                elif selected_page == "settings":
                    info("Rendering settings page", "app")
                    with LoggedOperation("render_settings_page", "app"):
                        render_settings_page()
                        
                elif selected_page == "analytics":
                    info("Rendering analytics page", "app")
                    with LoggedOperation("render_analytics_page", "app"):
                        render_analytics_page()
                        
                elif selected_page == "help":
                    info("Rendering help page", "app")
                    with LoggedOperation("render_help_page", "app"):
                        render_help_page()
                        
                elif selected_page == "debug":
                    info("Rendering debug page", "app")
                    with LoggedOperation("render_debug_page", "app"):
                        render_debug_page()
                
                elif selected_page == "auto_classes":
                    info("Rendering Auto Classes page", "app")
                    with LoggedOperation("render_auto_classes_page", "app"):
                        render_auto_classes_page()
                        
                elif selected_page == "web_server_inference":
                    info("Rendering Web Server Inference page", "app")
                    with LoggedOperation("render_web_server_inference_page", "app"):
                        render_web_server_inference_page()
                
                # Additional pages routing
                elif selected_page == "chat":
                    info("Rendering Chat page", "app")
                    with LoggedOperation("render_chat_page", "app"):
                        render_chat_page(self.model_manager)
                
                elif selected_page == "tokenizers":
                    info("Rendering Tokenizers page", "app")
                    with LoggedOperation("render_tokenizers_page", "app"):
                        render_tokenizers_page()
                
                elif selected_page == "tokenizer_summary":
                    info("Rendering Tokenizer Summary page", "app")
                    with LoggedOperation("render_tokenizer_summary_page", "app"):
                        render_tokenizer_summary_page()
                
                elif selected_page == "padding_truncation":
                    info("Rendering Padding & Truncation page", "app")
                    with LoggedOperation("render_padding_truncation_page", "app"):
                        render_padding_truncation_page()
                
                elif selected_page == "image_processors":
                    info("Rendering Image Processors page", "app")
                    with LoggedOperation("render_image_processors_page", "app"):
                        render_image_processors_page()
                
                elif selected_page == "video_processors":
                    info("Rendering Video Processors page", "app")
                    with LoggedOperation("render_video_processors_page", "app"):
                        render_video_processors_page()
                
                elif selected_page == "backbones":
                    info("Rendering Backbones page", "app")
                    with LoggedOperation("render_backbones_page", "app"):
                        render_backbones_page()
                
                elif selected_page == "processors":
                    info("Rendering Processors page", "app")
                    with LoggedOperation("render_processors_page", "app"):
                        render_processors_page()
                
                elif selected_page == "feature_extractors":
                    info("Rendering Feature Extractors page", "app")
                    with LoggedOperation("render_feature_extractors_page", "app"):
                        render_feature_extractors_page()
                
                elif selected_page == "pipeline":
                    info("Rendering Pipeline page", "app")
                    with LoggedOperation("render_pipeline_page", "app"):
                        render_pipeline_page()
                
                elif selected_page == "ml_apps":
                    info("Rendering ML Apps page", "app")
                    with LoggedOperation("render_ml_apps_page", "app"):
                        render_ml_apps_page()
                
                elif selected_page == "preprocessor_dashboard":
                    info("Rendering Preprocessor Dashboard", "app")
                    with LoggedOperation("render_preprocessor_dashboard", "app"):
                        render_preprocessor_dashboard()
                
                elif selected_page == "debug_dashboard":
                    info("Rendering Debug Dashboard", "app")
                    with LoggedOperation("render_debug_dashboard", "app"):
                        render_debug_dashboard()
                        
                else:
                    # Handle unknown page selection
                    warning(f"Unknown page selected: {selected_page}", "app")
                    st.error(f"Unknown page: {selected_page}")
                    self.render_home_page()  # Fallback to home page
                    
            except Exception as e:
                error(f"Error rendering page: {selected_page}", "app", e)
                st.error("An error occurred while loading the page. Please try again.")
                
                # Show error recovery options
                with st.expander("🛠️ Troubleshooting"):
                    st.write("If this error persists:")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Refresh Page"):
                            pass  # Streamlit will refresh automatically
                    with col2:
                        if st.button("🏠 Go to Home"):
                            st.session_state.current_page = "home"
                            # Streamlit will refresh automatically
            
            # Footer (currently commented out)
            # UIHelpers.display_footer()
            
            # Log successful page render
            log_session_event("page_rendered", page=selected_page)
            
        except Exception as e:
            # Critical application error - log and display
            error("Critical application error in main run loop", "app", e)
            st.error("🚨 Critical application error occurred. Please check logs and restart the application.")
            
            # Display error details for debugging
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                st.write("**Traceback:**")
                st.code(traceback.format_exc())
            
            raise


def main():
    """
    Main function to run the DurgasAI application.
    
    This is the entry point for the application. It:
    1. Creates the main DurgasAIApp instance
    2. Runs the application with comprehensive error handling
    3. Logs the application lifecycle events
    
    If any critical errors occur during startup, they are logged and re-raised.
    """
    info("=== DurgasAI Application Starting ===", "app")
    
    try:
        # Create the main application instance
        debug("Creating DurgasAIApp instance", "app")
        app = DurgasAIApp()
        
        # Run the application
        info("Starting application main loop", "app")
        app.run()
        
        info("=== DurgasAI Application Running Successfully ===", "app")
        
    except Exception as e:
        # Critical startup error
        error("=== CRITICAL: DurgasAI Application Failed to Start ===", "app", e)
        
        # Try to display error in Streamlit if possible
        try:
            st.error("🚨 **Critical Startup Error**")
            st.error("The application failed to start. Please check the logs for details.")
            with st.expander("🔍 Startup Error Details"):
                st.code(f"Error Type: {type(e).__name__}")
                st.code(f"Error Message: {str(e)}")
                st.code("Traceback:")
                st.code(traceback.format_exc())
        except:
            # Streamlit not available, print to console
            print(f"CRITICAL STARTUP ERROR: {e}")
            print(traceback.format_exc())
        
        # Re-raise the exception to stop execution
        raise


if __name__ == "__main__":
    # """
    # Application entry point.
    
    # When this script is run directly (not imported), it starts the DurgasAI application.
    # This includes:
    # - Environment setup (via suppress_warnings)
    # - Logging initialization (via utils.logger)
    # - Main application execution
    # """
    # Import traceback for error handling in main()
    import traceback
    
    # Log the application start
    info("=== DurgasAI Application Entry Point ===", "app", 
         python_version=sys.version,
         streamlit_version=st.__version__ if hasattr(st, '__version__') else "unknown",
         working_directory=str(Path.cwd()))
    
    main()
