"""
Service Manager for DurgasAI.

Centralized service coordination and dependency injection for the application.
This manager handles the initialization, coordination, and lifecycle management
of all application services.

Architecture:
- Singleton pattern for global service access
- Dependency injection and service coordination
- Lazy loading with proper initialization order
- Service health monitoring and status tracking
- Graceful shutdown and cleanup coordination

Key Features:
- Service initialization with proper dependency order
- Global service access point
- Service health monitoring
- Graceful shutdown coordination
- Error handling and recovery
- Service status tracking and reporting
"""

import asyncio
from typing import Dict, Any, Optional, Type
from datetime import datetime
from pathlib import Path
import sys
import threading
import atexit

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import debug, info, warning, error, LoggedOperation
from .config_service import ConfigService
from .model_service import ModelService
from .chat_service import ChatService
from .analytics_service import AnalyticsService


class ServiceManager:
    """
    Centralized service manager for DurgasAI application.
    
    This class manages the lifecycle of all application services,
    handles dependency injection, and provides a centralized access point.
    
    Singleton pattern ensures only one instance exists across the application.
    """
    
    _instance: Optional['ServiceManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ServiceManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the service manager."""
        if hasattr(self, '_initialized'):
            return
        
        debug("Initializing ServiceManager", "service_manager")
        
        # Service instances
        self._config_service: Optional[ConfigService] = None
        self._model_service: Optional[ModelService] = None
        self._chat_service: Optional[ChatService] = None
        self._analytics_service: Optional[AnalyticsService] = None
        
        # Service status tracking
        self._service_status: Dict[str, Dict[str, Any]] = {}
        self._initialization_order = [
            'config_service',
            'analytics_service', 
            'model_service',
            'chat_service'
        ]
        
        # Manager state
        self._initialized = False
        self._shutting_down = False
        
        # Register shutdown handler
        atexit.register(self.shutdown)
        
        info("ServiceManager initialized", "service_manager")
    
    async def initialize_services(self) -> bool:
        """
        Initialize all services in the proper dependency order.
        
        Returns:
            bool: True if all services initialized successfully
        """
        debug("Starting service initialization", "service_manager")
        
        with LoggedOperation("service_initialization", "service_manager"):
            try:
                success = True
                
                for service_name in self._initialization_order:
                    service_success = await self._initialize_service(service_name)
                    if not service_success:
                        error(f"Failed to initialize service: {service_name}", "service_manager")
                        success = False
                        break
                    
                    debug(f"Service initialized successfully: {service_name}", "service_manager")
                
                if success:
                    self._initialized = True
                    info("All services initialized successfully", "service_manager")
                else:
                    error("Service initialization failed", "service_manager")
                
                return success
                
            except Exception as e:
                error("Exception during service initialization", "service_manager", error_obj=e)
                return False
    
    async def _initialize_service(self, service_name: str) -> bool:
        """Initialize a specific service."""
        debug(f"Initializing service: {service_name}", "service_manager")
        
        start_time = datetime.now()
        
        try:
            if service_name == 'config_service':
                self._config_service = ConfigService()
                
            elif service_name == 'analytics_service':
                self._analytics_service = AnalyticsService(self._config_service)
                
            elif service_name == 'model_service':
                self._model_service = ModelService(self._config_service)
                
            elif service_name == 'chat_service':
                self._chat_service = ChatService(self._config_service, self._model_service)
            
            else:
                error(f"Unknown service: {service_name}", "service_manager")
                return False
            
            # Track service status
            initialization_time = (datetime.now() - start_time).total_seconds()
            self._service_status[service_name] = {
                'status': 'initialized',
                'initialized_at': start_time.isoformat(),
                'initialization_time': initialization_time,
                'last_health_check': datetime.now().isoformat()
            }
            
            info(f"Service {service_name} initialized in {initialization_time:.2f}s", "service_manager")
            return True
            
        except Exception as e:
            error(f"Failed to initialize {service_name}", "service_manager", error_obj=e)
            
            self._service_status[service_name] = {
                'status': 'error',
                'error_message': str(e),
                'failed_at': datetime.now().isoformat()
            }
            
            return False
    
    def get_config_service(self) -> Optional[ConfigService]:
        """Get the configuration service."""
        if not self._initialized:
            warning("ServiceManager not initialized", "service_manager")
        return self._config_service
    
    def get_model_service(self) -> Optional[ModelService]:
        """Get the model service."""
        if not self._initialized:
            warning("ServiceManager not initialized", "service_manager")
        return self._model_service
    
    def get_chat_service(self) -> Optional[ChatService]:
        """Get the chat service."""
        if not self._initialized:
            warning("ServiceManager not initialized", "service_manager")
        return self._chat_service
    
    def get_analytics_service(self) -> Optional[AnalyticsService]:
        """Get the analytics service."""
        if not self._initialized:
            warning("ServiceManager not initialized", "service_manager")
        return self._analytics_service
    
    def is_initialized(self) -> bool:
        """Check if all services are initialized."""
        return self._initialized
    
    def get_service_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services."""
        debug("Retrieving service status", "service_manager")
        
        # Update health check timestamps
        current_time = datetime.now().isoformat()
        for service_name in self._service_status:
            if self._service_status[service_name]['status'] == 'initialized':
                self._service_status[service_name]['last_health_check'] = current_time
        
        return self._service_status.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        debug("Performing service health check", "service_manager")
        
        health_status = {
            'overall_status': 'healthy',
            'services': {},
            'checked_at': datetime.now().isoformat()
        }
        
        for service_name in self._initialization_order:
            service_health = self._check_service_health(service_name)
            health_status['services'][service_name] = service_health
            
            if service_health['status'] != 'healthy':
                health_status['overall_status'] = 'degraded'
        
        debug("Health check completed", "service_manager", status=health_status['overall_status'])
        return health_status
    
    def _check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service."""
        try:
            service = getattr(self, f'_{service_name}', None)
            
            if service is None:
                return {
                    'status': 'not_initialized',
                    'message': 'Service not initialized'
                }
            
            # Basic health check - service exists and is accessible
            # More sophisticated health checks could be added per service
            return {
                'status': 'healthy',
                'message': 'Service is operational',
                'checked_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'error_at': datetime.now().isoformat()
            }
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        debug(f"Restarting service: {service_name}", "service_manager")
        
        try:
            # Mark service as restarting
            self._service_status[service_name] = {
                'status': 'restarting',
                'restarting_at': datetime.now().isoformat()
            }
            
            # Reinitialize the service
            success = asyncio.run(self._initialize_service(service_name))
            
            if success:
                info(f"Service restarted successfully: {service_name}", "service_manager")
            else:
                error(f"Failed to restart service: {service_name}", "service_manager")
            
            return success
            
        except Exception as e:
            error(f"Exception during service restart: {service_name}", "service_manager", error_obj=e)
            return False
    
    def shutdown(self) -> None:
        """Shutdown all services gracefully."""
        if self._shutting_down:
            return
        
        debug("Starting service shutdown", "service_manager")
        self._shutting_down = True
        
        with LoggedOperation("service_shutdown", "service_manager"):
            try:
                # Shutdown services in reverse order
                shutdown_order = list(reversed(self._initialization_order))
                
                for service_name in shutdown_order:
                    self._shutdown_service(service_name)
                
                info("All services shutdown completed", "service_manager")
                
            except Exception as e:
                error("Exception during service shutdown", "service_manager", error_obj=e)
    
    def _shutdown_service(self, service_name: str) -> None:
        """Shutdown a specific service."""
        debug(f"Shutting down service: {service_name}", "service_manager")
        
        try:
            service = getattr(self, f'_{service_name}', None)
            
            if service is None:
                debug(f"Service not initialized, skipping shutdown: {service_name}", "service_manager")
                return
            
            # Call shutdown method if available
            if hasattr(service, 'shutdown'):
                service.shutdown()
                debug(f"Service shutdown method called: {service_name}", "service_manager")
            
            # Update status
            self._service_status[service_name] = {
                'status': 'shutdown',
                'shutdown_at': datetime.now().isoformat()
            }
            
            info(f"Service shutdown completed: {service_name}", "service_manager")
            
        except Exception as e:
            error(f"Exception during {service_name} shutdown", "service_manager", error_obj=e)
    
    @classmethod
    def get_instance(cls) -> 'ServiceManager':
        """Get the singleton instance."""
        return cls()
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        debug("Retrieving service metrics", "service_manager")
        
        metrics = {
            'service_count': len(self._initialization_order),
            'initialized_services': len([
                s for s in self._service_status.values() 
                if s.get('status') == 'initialized'
            ]),
            'failed_services': len([
                s for s in self._service_status.values() 
                if s.get('status') == 'error'
            ]),
            'total_initialization_time': sum([
                s.get('initialization_time', 0) 
                for s in self._service_status.values()
            ]),
            'manager_status': {
                'initialized': self._initialized,
                'shutting_down': self._shutting_down
            }
        }
        
        # Add analytics from analytics service if available
        if self._analytics_service:
            try:
                analytics_metrics = self._analytics_service.get_real_time_metrics()
                metrics['analytics'] = analytics_metrics
            except Exception as e:
                debug("Failed to get analytics metrics", "service_manager", error_obj=e)
        
        return metrics
