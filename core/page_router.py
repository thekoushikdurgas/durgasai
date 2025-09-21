"""
Page Router for DurgasAI Application.

Handles page routing, navigation logic, and page lifecycle management
separate from UI rendering concerns.
"""

from typing import Dict, Any, Callable, Optional
from pathlib import Path
import sys

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, log_user_action
from .application_state import ApplicationState


class PageRouter:
    """
    Manages page routing and navigation for the DurgasAI application.
    
    This class handles:
    - Page registration and routing logic
    - Navigation state management
    - Page lifecycle events
    - Navigation analytics
    """
    
    def __init__(self):
        """Initialize the page router."""
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.current_page: Optional[str] = None
        debug("PageRouter initialized", "router")
    
    def register_page(self, page_id: str, page_config: Dict[str, Any]) -> None:
        """
        Register a page with the router.
        
        Args:
            page_id: Unique identifier for the page
            page_config: Page configuration including title, icon, render function, etc.
        """
        required_fields = ['title', 'render_function']
        for field in required_fields:
            if field not in page_config:
                raise ValueError(f"Page config missing required field: {field}")
        
        self.pages[page_id] = {
            'id': page_id,
            'title': page_config['title'],
            'icon': page_config.get('icon', '📄'),
            'render_function': page_config['render_function'],
            'requires_model': page_config.get('requires_model', False),
            'description': page_config.get('description', ''),
            'order': page_config.get('order', 999),
            'enabled': page_config.get('enabled', True)
        }
        
        debug(f"Page registered: {page_id}", "router", title=page_config['title'])
    
    def get_navigation_items(self) -> Dict[str, str]:
        """Get navigation items for UI rendering."""
        enabled_pages = {
            f"{config['icon']} {config['title']}": page_id 
            for page_id, config in self.pages.items() 
            if config['enabled']
        }
        
        # Sort by order
        sorted_pages = dict(sorted(
            enabled_pages.items(),
            key=lambda x: self.pages[x[1]]['order']
        ))
        
        return sorted_pages
    
    def navigate_to(self, page_id: str) -> bool:
        """
        Navigate to a specific page.
        
        Args:
            page_id: ID of the page to navigate to
            
        Returns:
            bool: True if navigation successful, False otherwise
        """
        if page_id not in self.pages:
            debug(f"Page not found: {page_id}", "router")
            return False
        
        page_config = self.pages[page_id]
        
        # Check if page requires model and model is loaded
        if page_config['requires_model'] and not ApplicationState.is_model_loaded():
            debug(f"Page {page_id} requires model but none loaded", "router")
            return False
        
        previous_page = self.current_page
        self.current_page = page_id
        
        # Update application state
        ApplicationState.set('current_page', page_id)
        
        # Log navigation
        log_user_action("page_navigation",
            selected_page=page_id,
            previous_page=previous_page,
            page_title=page_config['title'])
        
        info(f"Navigated to page: {page_config['title']}", "router")
        return True
    
    def render_current_page(self) -> None:
        """Render the current page."""
        if not self.current_page or self.current_page not in self.pages:
            debug("No valid current page, defaulting to home", "router")
            self.navigate_to('home')
            return
        
        page_config = self.pages[self.current_page]
        
        try:
            debug(f"Rendering page: {self.current_page}", "router")
            page_config['render_function']()
            
        except Exception as e:
            debug(f"Error rendering page {self.current_page}: {e}", "router")
            raise
    
    def get_page_info(self, page_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific page."""
        return self.pages.get(page_id)
    
    def get_current_page_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current page."""
        if self.current_page:
            return self.pages.get(self.current_page)
        return None
