"""
Main Application Controller for DurgasAI.

This module contains the core application orchestration logic,
separated from UI concerns and routing. It acts as the central coordinator
for all application services and manages the application lifecycle.

Architecture:
- Initializes and coordinates core services (ModelManager, PageRouter)
- Manages application configuration and state
- Handles service dependencies and injection
- Provides centralized error handling and recovery
- Orchestrates the application startup sequence

Dependencies:
- utils.config: Application configuration management
- utils.logger: Centralized logging system
- utils.model_manager: AI model management
- application_state: Global state management
- page_router: Navigation and page routing
"""

from typing import Optional
from pathlib import Path
import sys

# Add utils to path for proper module resolution
# This ensures utils modules can be imported from any location
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from utils.logger import debug, info, warning, error, log_session_event, LoggedOperation, log_user_action
from utils.model_manager import ModelManager
from .application_state import ApplicationState
from .page_router import PageRouter


class DurgasAIController:
    """
    Main application controller that orchestrates the DurgasAI system.
    
    This class is responsible for:
    - Application initialization and configuration
    - Service coordination and dependency injection
    - Application lifecycle management
    - Error handling and recovery
    """
    
    def __init__(self):
        """
        Initialize the application controller.
        
        This constructor sets up the core application services and initializes
        the application in the correct order to ensure proper dependency resolution.
        
        Raises:
            Exception: If critical initialization steps fail
        """
        debug("Starting DurgasAI Controller initialization", "controller")
        
        # Initialize core service references
        # These will be populated during the initialization process
        self.model_manager: Optional[ModelManager] = None
        self.page_router: Optional[PageRouter] = None
        
        debug("Core service references initialized", "controller")
        
        # Start the application initialization sequence
        # This must be called last to ensure all instance variables are set
        self._initialize_application()
        
        debug("DurgasAI Controller initialization completed successfully", "controller")
    
    def _initialize_application(self) -> None:
        """
        Initialize core application components in the correct order.
        
        This method follows a specific initialization sequence to ensure
        proper dependency resolution and error handling:
        1. Load configuration files
        2. Initialize application state
        3. Initialize model manager
        4. Initialize page router and register pages
        5. Mark application as ready
        
        Raises:
            Exception: If any critical initialization step fails
        """
        debug("Starting application initialization sequence", "controller")
        
        try:
            with LoggedOperation("app_initialization", "controller"):
                # Step 1: Load configuration files
                # This must happen first as other components depend on configuration
                debug("Step 1/5: Loading HuggingFace configuration", "controller")
                hf_config = Config.load_huggingface_config()
                debug(f"HuggingFace config loaded with cache_dir: {hf_config.get('cache_dir', 'default')}", "controller")
                info("Configuration loaded successfully", "controller")
                
                # Step 2: Initialize application state
                # This sets up the global state management system
                debug("Step 2/5: Initializing application state", "controller")
                ApplicationState.initialize()
                debug("Application state initialization completed", "controller")
                
                # Step 3: Initialize model manager
                # This sets up AI model management capabilities
                debug("Step 3/5: Initializing model manager", "controller")
                self.model_manager = ModelManager()
                debug("Model manager instance created", "controller")
                info("Model manager initialized successfully", "controller")
                
                # Step 4: Initialize page router and register pages
                # This sets up navigation and page management
                debug("Step 4/5: Initializing page router", "controller")
                self.page_router = PageRouter()
                debug("Page router instance created", "controller")
                
                debug("Registering application pages", "controller")
                self._register_pages()
                debug("Page registration completed", "controller")
                
                # Step 5: Mark application as fully initialized
                debug("Step 5/5: Finalizing application initialization", "controller")
                ApplicationState.set('app_initialized', True)
                log_session_event("app_controller_initialized")
                
                info("DurgasAI Controller initialization sequence completed successfully", "controller")
                
        except Exception as e:
            error("Critical failure during application initialization", "controller", e)
            # Log additional context for debugging
            debug(f"Initialization failed at step with error: {str(e)}", "controller")
            raise
    
    def _register_pages(self) -> None:
        """
        Register all application pages with the page router with comprehensive logging.
        
        This method imports and registers all page render functions with the
        page router, setting up the navigation system. Pages are registered
        with their configuration including title, icon, render function, and
        any special requirements. Includes detailed logging for debugging and
        error handling for missing page modules.
        
        Features:
        - Dynamic page import with error handling
        - Page configuration validation
        - Registration status tracking
        - Error recovery for missing pages
        - User action logging for analytics
        
        Logging:
        - Page import status and errors
        - Registration process tracking
        - Configuration validation results
        - Error handling for missing modules
        - Performance metrics for registration
        
        Note: Page imports are done here to avoid circular import issues
        and to ensure all dependencies are properly initialized.
        """
        debug("Starting comprehensive page registration process", "controller")
        
        try:
            with LoggedOperation("page_registration", "controller"):
                # Import page render functions with comprehensive error handling
                debug("Importing page render functions with error handling", "controller")
                
                imported_pages = {}
                import_errors = []
                
                # Define page imports with error handling
                page_imports = [
                    ("page.home_page", "render_home_page"),
                    ("page.chat_page", "render_chat_page"),
                    ("page.settings_page", "render_settings_page"),
                    ("page.analytics_page", "render_analytics_page"),
                    ("page.help_page", "render_help_page"),
                    ("page.debug_page", "render_debug_page"),
                    ("page.model_catalog_page", "render_model_catalog_page"),
                    ("page.tokenizers_page", "render_tokenizers_page"),
                    ("page.image_processors_page", "render_image_processors_page"),
                    ("page.video_processors_page", "render_video_processors_page"),
                    ("page.backbones_page", "render_backbones_page"),
                    ("page.feature_extractors_page", "render_feature_extractors_page"),
                    ("page.processors_page", "render_processors_page"),
                    ("page.tokenizer_summary_page", "render_tokenizer_summary_page"),
                    ("page.padding_truncation_page", "render_padding_truncation_page"),
                    ("page.pipeline_page", "render_pipeline_page"),
                    ("page.ml_apps_page", "render_ml_apps_page"),
                    ("page.preprocessor_dashboard", "render_preprocessor_dashboard")
                ]
                
                debug(f"Attempting to import {len(page_imports)} page modules", "controller")
                
                # Import each page module with error handling
                for module_name, function_name in page_imports:
                    try:
                        debug(f"Importing {module_name}.{function_name}", "controller")
                        
                        # Dynamic import with error handling
                        module = __import__(module_name, fromlist=[function_name])
                        render_function = getattr(module, function_name)
                        
                        imported_pages[function_name] = render_function
                        debug(f"Successfully imported {module_name}.{function_name}", "controller")
                        
                    except ImportError as e:
                        error_msg = f"Failed to import {module_name}: {str(e)}"
                        warning(error_msg, "controller", module=module_name, error=str(e))
                        import_errors.append(error_msg)
                        
                        # Continue with other imports
                        continue
                        
                    except AttributeError as e:
                        error_msg = f"Function {function_name} not found in {module_name}: {str(e)}"
                        warning(error_msg, "controller", module=module_name, function=function_name, error=str(e))
                        import_errors.append(error_msg)
                        
                        # Continue with other imports
                        continue
                        
                    except Exception as e:
                        error_msg = f"Unexpected error importing {module_name}: {str(e)}"
                        error(error_msg, "controller", e, module=module_name)
                        import_errors.append(error_msg)
                        
                        # Continue with other imports
                        continue
                
                debug("Page import process completed", "controller",
                      successful_imports=len(imported_pages),
                      failed_imports=len(import_errors),
                      import_errors=import_errors)
                
                # Log import results for analytics
                log_user_action("page_imports_completed",
                              successful_count=len(imported_pages),
                              failed_count=len(import_errors),
                              total_attempted=len(page_imports))
                
                if import_errors:
                    warning(f"Some page imports failed: {len(import_errors)} errors", "controller",
                           errors=import_errors)
                
                # Validate that we have at least the core pages
                required_pages = ['render_home_page', 'render_chat_page']
                missing_required = [page for page in required_pages if page not in imported_pages]
                
                if missing_required:
                    error(f"Critical pages missing: {missing_required}", "controller")
                    raise ImportError(f"Required pages could not be imported: {missing_required}")
                
                debug("Core page validation passed", "controller")
                
        except Exception as e:
            error("Critical error during page imports", "controller", e)
            raise
        
        # Continue with page registration using imported functions
        debug("Page imports completed, proceeding with registration", "controller")
        
        # Now register pages using the imported functions
        debug("Starting page registration with imported functions", "controller")
        
        # Register pages with router using imported functions
        try:
            with LoggedOperation("page_configuration_setup", "controller"):
                pages_config = [
                    {
                        'id': 'home',
                        'title': 'Home',
                        'icon': '🏠',
                        'render_function': imported_pages.get('render_home_page'),
                        'order': 1
                    },
                    {
                        'id': 'chat',
                        'title': 'AI Agent',
                        'icon': '🤖',
                        'render_function': lambda: imported_pages['render_chat_page'](self.model_manager) if 'render_chat_page' in imported_pages else None,
                        'requires_model': False,
                        'order': 2
                    },
                    {
                        'id': 'model_catalog',
                        'title': 'Model Catalog',
                        'icon': '📚',
                        'render_function': imported_pages.get('render_model_catalog_page'),
                        'order': 3
                    },
                    {
                        'id': 'preprocessor_dashboard',
                        'title': 'Preprocessor Dashboard',
                        'icon': '🔧',
                        'render_function': imported_pages.get('render_preprocessor_dashboard'),
                        'order': 4
                    },
                    {
                        'id': 'tokenizers',
                        'title': 'Tokenizers',
                        'icon': '🔤',
                        'render_function': imported_pages.get('render_tokenizers_page'),
                        'order': 5
                    },
                    {
                        'id': 'image_processors',
                        'title': 'Image Processors',
                        'icon': '🖼️',
                        'render_function': imported_pages.get('render_image_processors_page'),
                        'order': 6
                    },
                    {
                        'id': 'video_processors',
                        'title': 'Video Processors',
                        'icon': '🎥',
                        'render_function': imported_pages.get('render_video_processors_page'),
                        'order': 7
                    },
                    {
                        'id': 'backbones',
                        'title': 'Backbones',
                        'icon': '🏗️',
                        'render_function': imported_pages.get('render_backbones_page'),
                        'order': 8
                    },
                    {
                        'id': 'feature_extractors',
                        'title': 'Feature Extractors',
                        'icon': '🎵',
                        'render_function': imported_pages.get('render_feature_extractors_page'),
                        'order': 9
                    },
                    {
                        'id': 'processors',
                        'title': 'Processors',
                        'icon': '🔄',
                        'render_function': imported_pages.get('render_processors_page'),
                        'order': 10
                    },
                    {
                        'id': 'tokenizer_summary',
                        'title': 'Tokenizer Summary',
                        'icon': '📝',
                        'render_function': imported_pages.get('render_tokenizer_summary_page'),
                        'order': 11
                    },
                    {
                        'id': 'padding_truncation',
                        'title': 'Padding & Truncation',
                        'icon': '📏',
                        'render_function': imported_pages.get('render_padding_truncation_page'),
                        'order': 12
                    },
                    {
                        'id': 'pipeline',
                        'title': 'Pipeline',
                        'icon': '🔄',
                        'render_function': imported_pages.get('render_pipeline_page'),
                        'order': 13
                    },
                    {
                        'id': 'ml_apps',
                        'title': 'ML Apps',
                        'icon': '🚀',
                        'render_function': imported_pages.get('render_ml_apps_page'),
                        'order': 14
                    },
                    {
                        'id': 'settings',
                        'title': 'Settings',
                        'icon': '⚙️',
                        'render_function': imported_pages.get('render_settings_page'),
                        'order': 15
                    },
                    {
                        'id': 'analytics',
                        'title': 'Analytics',
                        'icon': '📊',
                        'render_function': imported_pages.get('render_analytics_page'),
                        'order': 16
                    },
                    {
                        'id': 'help',
                        'title': 'Help',
                        'icon': '❓',
                        'render_function': imported_pages.get('render_help_page'),
                        'order': 17
                    },
                    {
                        'id': 'debug',
                        'title': 'Debug',
                        'icon': '🔧',
                        'render_function': imported_pages.get('render_debug_page'),
                        'order': 18
                    }
                ]
                
                # Filter out pages with None render functions
                valid_pages = [page for page in pages_config if page['render_function'] is not None]
                invalid_pages = [page for page in pages_config if page['render_function'] is None]
                
                debug("Page configuration validation completed", "controller",
                      total_pages=len(pages_config),
                      valid_pages=len(valid_pages),
                      invalid_pages=len(invalid_pages),
                      invalid_page_ids=[page['id'] for page in invalid_pages])
                
                if invalid_pages:
                    warning(f"Some pages could not be configured: {len(invalid_pages)} pages", "controller",
                           invalid_page_ids=[page['id'] for page in invalid_pages])
                
                # Log page configuration for analytics
                log_user_action("page_configuration_completed",
                              total_pages=len(pages_config),
                              valid_pages=len(valid_pages),
                              invalid_pages=len(invalid_pages))
                
        except Exception as e:
            error("Critical error in page configuration setup", "controller", e)
            raise
        
        # Register each valid page with the router
        debug(f"Registering {len(valid_pages)} valid pages with router", "controller")
        
        registration_success_count = 0
        registration_errors = []
        
        for page_config in valid_pages:
            try:
                page_id = page_config['id']
                page_title = page_config.get('title', 'Unknown')
                
                debug(f"Registering page: {page_id} - {page_title}", "controller")
                
                # Create a copy of config without 'id' for registration
                registration_config = {k: v for k, v in page_config.items() if k != 'id'}
                
                self.page_router.register_page(page_id, registration_config)
                registration_success_count += 1
                
                debug(f"Successfully registered page: {page_id}", "controller")
                
            except Exception as e:
                error_msg = f"Failed to register page {page_config.get('id', 'unknown')}: {str(e)}"
                error(error_msg, "controller", e,
                      page_id=page_config.get('id'),
                      page_title=page_config.get('title'))
                registration_errors.append(error_msg)
                continue
        
        debug("Page registration process completed", "controller",
              total_valid_pages=len(valid_pages),
              successful_registrations=registration_success_count,
              failed_registrations=len(registration_errors),
              registration_errors=registration_errors)
        
        # Log registration results for analytics
        log_user_action("page_registration_completed",
                      total_valid_pages=len(valid_pages),
                      successful_registrations=registration_success_count,
                      failed_registrations=len(registration_errors))
        
        if registration_errors:
            warning(f"Some page registrations failed: {len(registration_errors)} errors", "controller",
                   errors=registration_errors)
        
        info(f"Successfully registered {registration_success_count}/{len(valid_pages)} pages with the router", "controller")
        
        debug("Page registration process completed successfully", "controller")
    
    def get_model_manager(self) -> ModelManager:
        """
        Get the model manager instance.
        
        Returns:
            ModelManager: The initialized model manager instance
            
        Note:
            This method provides access to the model manager for other
            components that need to interact with AI models.
        """
        debug("Providing model manager instance to caller", "controller")
        return self.model_manager
    
    def get_page_router(self) -> PageRouter:
        """
        Get the page router instance.
        
        Returns:
            PageRouter: The initialized page router instance
            
        Note:
            This method provides access to the page router for navigation
            and page management operations.
        """
        debug("Providing page router instance to caller", "controller")
        return self.page_router
    
    def shutdown(self) -> None:
        """
        Shutdown the application gracefully.
        
        This method performs cleanup operations including:
        - Clearing sensitive data from application state
        - Logging shutdown event
        - Ensuring proper resource cleanup
        
        Note:
            This method should be called when the application is being
            terminated to ensure proper cleanup.
        """
        debug("Initiating graceful application shutdown", "controller")
        info("Starting DurgasAI Controller shutdown sequence", "controller")
        
        # Clear sensitive data from application state
        debug("Clearing sensitive data from application state", "controller")
        ApplicationState.set('api_token', '')
        debug("Sensitive data cleared", "controller")
        
        # Log shutdown event for analytics
        log_session_event("app_controller_shutdown")
        debug("Shutdown event logged", "controller")
        
        info("DurgasAI Controller shutdown sequence completed successfully", "controller")
