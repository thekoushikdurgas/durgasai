"""
Core application modules for DurgasAI.

This package contains the core application logic separated from UI concerns:
- app_controller: Main application orchestration
- page_router: Page routing and navigation logic
- application_state: Global application state management
"""

from .app_controller import DurgasAIController
from .page_router import PageRouter
from .application_state import ApplicationState

__all__ = ['DurgasAIController', 'PageRouter', 'ApplicationState']
