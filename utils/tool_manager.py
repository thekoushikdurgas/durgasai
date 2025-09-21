"""
Tool Manager for DurgasAI
Handles tool execution, validation, and logging with comprehensive error handling.
"""

import json
import os
import sys
import importlib.util
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import traceback

from .logger import debug, info, warning, error, log_tool_execution, LoggedOperation


class ToolManager:
    """
    Manages tool execution with comprehensive logging and error handling.
    
    This class provides:
    - Tool discovery and loading from JSON configurations
    - Safe tool execution with input validation
    - Comprehensive logging of tool operations
    - Error handling and recovery
    - Performance monitoring
    - Security validation
    """
    
    def __init__(self, tools_directory: str = "output/tools"):
        """
        Initialize the ToolManager.
        
        Args:
            tools_directory (str): Directory containing tool configurations and code
        """
        self.tools_directory = Path(tools_directory)
        self.code_directory = self.tools_directory / "code"
        self.loaded_tools = {}
        self.tool_configs = {}
        self.execution_stats = {}
        
        debug("Initializing ToolManager", "tools",
            tools_directory=str(self.tools_directory),
            code_directory=str(self.code_directory))
        
        # Load available tools
        self.load_available_tools()
        
        info(f"ToolManager initialized with {len(self.loaded_tools)} tools", "tools", {
            "available_tools": list(self.loaded_tools.keys()),
            "tools_directory": str(self.tools_directory)
        })
    
    def load_available_tools(self):
        """
        Load all available tools from the tools directory.
        
        This method:
        1. Scans for JSON tool configuration files
        2. Validates tool configurations
        3. Loads corresponding Python code files
        4. Registers tools for execution
        """
        debug("Loading available tools", "tools")
        
        if not self.tools_directory.exists():
            warning(f"Tools directory not found: {self.tools_directory}", "tools")
            return
        
        # Find all JSON tool configuration files
        tool_config_files = list(self.tools_directory.glob("*.json"))
        debug(f"Found {len(tool_config_files)} tool configuration files", "tools")
        
        for config_file in tool_config_files:
            try:
                # Load tool configuration
                with open(config_file, 'r', encoding='utf-8') as f:
                    tool_config = json.load(f)
                
                tool_name = tool_config.get("name") or config_file.stem
                
                debug(f"Loading tool: {tool_name}", "tools",
                    config_file=config_file.name,
                    status=tool_config.get("status", "unknown"))
                
                # Validate tool configuration
                if not self._validate_tool_config(tool_config, tool_name):
                    continue
                
                # Load tool code
                if self._load_tool_code(tool_config, tool_name):
                    self.tool_configs[tool_name] = tool_config
                    info(f"Tool loaded successfully: {tool_name}", "tools")
                
            except Exception as e:
                error(f"Error loading tool from {config_file.name}", "tools", e)
        
        info(f"Tool loading completed: {len(self.loaded_tools)} tools available", "tools", {
            "loaded_tools": list(self.loaded_tools.keys())
        })
    
    def _validate_tool_config(self, tool_config: Dict[str, Any], tool_name: str) -> bool:
        """
        Validate tool configuration for required fields and security.
        
        Args:
            tool_config (Dict[str, Any]): Tool configuration dictionary
            tool_name (str): Name of the tool
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        required_fields = ["function_name", "code"]
        
        for field in required_fields:
            if field not in tool_config:
                warning(f"Tool {tool_name} missing required field: {field}", "tools")
                return False
        
        # Check security level
        security_level = tool_config.get("security_level", "medium")
        if security_level == "disabled":
            debug(f"Tool {tool_name} is disabled", "tools")
            return False
        
        # Validate status
        status = tool_config.get("status", "unknown")
        if status != "enabled":
            debug(f"Tool {tool_name} is not enabled (status: {status})", "tools")
            return False
        
        debug(f"Tool configuration validated: {tool_name}", "tools",
            security_level=security_level,
            status=status)
        
        return True
    
    def _load_tool_code(self, tool_config: Dict[str, Any], tool_name: str) -> bool:
        """
        Load tool code from Python file.
        
        Args:
            tool_config (Dict[str, Any]): Tool configuration
            tool_name (str): Name of the tool
            
        Returns:
            bool: True if code loaded successfully, False otherwise
        """
        code_file = tool_config.get("code")
        if not code_file:
            warning(f"No code file specified for tool: {tool_name}", "tools")
            return False
        
        code_path = self.code_directory / code_file
        
        if not code_path.exists():
            warning(f"Code file not found for tool {tool_name}: {code_path}", "tools")
            return False
        
        try:
            # Load the Python module
            spec = importlib.util.spec_from_file_location(tool_name, code_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the function
            function_name = tool_config.get("function_name")
            if hasattr(module, function_name):
                self.loaded_tools[tool_name] = getattr(module, function_name)
                debug(f"Tool function loaded: {tool_name}.{function_name}", "tools")
                return True
            else:
                warning(f"Function {function_name} not found in {code_file}", "tools")
                return False
        
        except Exception as e:
            error(f"Error loading code for tool {tool_name}", "tools", e,
                code_file=str(code_path),
                function_name=tool_config.get("function_name"))
            return False
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool with comprehensive logging and error handling.
        
        Args:
            tool_name (str): Name of the tool to execute
            **kwargs: Tool input parameters
            
        Returns:
            Dict[str, Any]: Tool execution result with metadata
        """
        start_time = time.time()
        
        info(f"Executing tool: {tool_name}", "tools",
            input_parameters=list(kwargs.keys()),
            parameter_count=len(kwargs))
        
        # Validate tool availability
        if tool_name not in self.loaded_tools:
            error_msg = f"Tool not found: {tool_name}"
            warning(error_msg, "tools",
                available_tools=list(self.loaded_tools.keys()))
            
            return {
                "success": False,
                "error": error_msg,
                "execution_time": 0,
                "tool_name": tool_name
            }
        
        # Initialize execution stats if needed
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0,
                "average_execution_time": 0
            }
        
        try:
            with LoggedOperation(f"tool_execution_{tool_name}", "tools", extra_data=kwargs):
                # Execute the tool function
                debug(f"Calling tool function: {tool_name}", "tools",
                    input_data={k: str(v)[:100] for k, v in kwargs.items()})
                
                tool_function = self.loaded_tools[tool_name]
                result = tool_function(**kwargs)
                
                execution_time = time.time() - start_time
                
                # Update execution statistics
                stats = self.execution_stats[tool_name]
                stats["total_executions"] += 1
                stats["successful_executions"] += 1
                stats["total_execution_time"] += execution_time
                stats["average_execution_time"] = stats["total_execution_time"] / stats["total_executions"]
                
                # Log successful execution
                info(f"Tool executed successfully: {tool_name} in {execution_time:.3f}s", "tools",
                    execution_time=execution_time,
                    result_type=type(result).__name__,
                    result_length=len(str(result)) if result else 0,
                    execution_stats=stats)
                
                # Log tool execution for analytics
                log_tool_execution(tool_name, kwargs, {"result": str(result)[:200]})
                
                return {
                    "success": True,
                    "result": result,
                    "execution_time": execution_time,
                    "tool_name": tool_name,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "execution_stats": stats
                    }
                }
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update failure statistics
            stats = self.execution_stats[tool_name]
            stats["total_executions"] += 1
            stats["failed_executions"] += 1
            stats["total_execution_time"] += execution_time
            stats["average_execution_time"] = stats["total_execution_time"] / stats["total_executions"]
            
            error_msg = f"Tool execution failed: {tool_name}: {str(e)}"
            error(error_msg, "tools", e,
                tool_name=tool_name,
                execution_time=execution_time,
                input_parameters=kwargs,
                execution_stats=stats)
            
            # Log failed tool execution
            log_tool_execution(tool_name, kwargs, error=e)
            
            return {
                "success": False,
                "error": error_msg,
                "execution_time": execution_time,
                "tool_name": tool_name,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(e).__name__,
                    "execution_stats": stats
                }
            }
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name (str): Name of the tool
            
        Returns:
            Optional[Dict[str, Any]]: Tool configuration and statistics
        """
        if tool_name not in self.tool_configs:
            return None
        
        config = self.tool_configs[tool_name].copy()
        config["execution_stats"] = self.execution_stats.get(tool_name, {})
        config["is_loaded"] = tool_name in self.loaded_tools
        
        return config
    
    def get_all_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available tools.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of tool names to their info
        """
        tools_info = {}
        
        for tool_name in self.tool_configs:
            tools_info[tool_name] = self.get_tool_info(tool_name)
        
        return tools_info
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get overall tool execution statistics.
        
        Returns:
            Dict[str, Any]: Execution statistics across all tools
        """
        total_executions = sum(stats.get("total_executions", 0) for stats in self.execution_stats.values())
        total_successes = sum(stats.get("successful_executions", 0) for stats in self.execution_stats.values())
        total_failures = sum(stats.get("failed_executions", 0) for stats in self.execution_stats.values())
        
        return {
            "total_tools": len(self.loaded_tools),
            "total_executions": total_executions,
            "successful_executions": total_successes,
            "failed_executions": total_failures,
            "success_rate": (total_successes / total_executions * 100) if total_executions > 0 else 0,
            "individual_stats": self.execution_stats
        }


# Global tool manager instance
_tool_manager = None

def get_tool_manager() -> ToolManager:
    """Get the global tool manager instance."""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager

def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool with the global tool manager."""
    return get_tool_manager().execute_tool(tool_name, **kwargs)

def get_available_tools() -> List[str]:
    """Get list of available tool names."""
    return list(get_tool_manager().loaded_tools.keys())

def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific tool."""
    return get_tool_manager().get_tool_info(tool_name)
