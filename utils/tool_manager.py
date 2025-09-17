"""
Enhanced Tool Manager for DurgasAI
Manages integration of DurgasAI tools with LangGraph chatbot
Enhanced with better validation, testing, and LangChain integration
"""

import json
import importlib
import os
import inspect
import time
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Union
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError
from utils.logging_utils import log_info, log_error, log_debug, log_warning
from models.tool_models import ToolDefinition, ToolExecution, ToolSecurityReport, ToolExecutionResult, validate_tool_code, generate_security_report
from models import create_tool_definition, create_execution_record


class ToolValidationError(Exception):
    """Custom exception for tool validation errors"""
    pass

class DurgasAIToolManager:
    """Enhanced DurgasAI tool manager for LangGraph integration"""
    
    def __init__(self, tools_directory: str = "output/tools"):
        self.tools_directory = Path(tools_directory)
        self.tools_cache = {}
        self.tool_functions = {}
        self.tool_metadata = {}
        self.execution_history = []
        self._load_tools()
        log_info(f"Tool manager initialized with {len(self.tools_cache)} tools", "tools")
    
    def _load_tools(self):
        """Load all available tools from the tools directory with enhanced error handling"""
        try:
            # Load tool definitions from JSON files
            for tool_file in self.tools_directory.glob("*.json"):
                if tool_file.name == "code":
                    continue  # Skip code directory
                
                try:
                    with open(tool_file, 'r', encoding='utf-8') as f:
                        tool_def = json.load(f)
                    
                    tool_name = tool_def.get("name")
                    if tool_name:
                        # Validate tool definition
                        validation_errors = self._validate_tool_definition(tool_def)
                        if validation_errors:
                            log_warning(f"Tool {tool_name} has validation issues: {validation_errors}", "tools")
                        
                        self.tools_cache[tool_name] = tool_def
                        
                        # Load corresponding Python function
                        self._load_tool_function(tool_name, tool_def)
                        
                        # Initialize metadata
                        self.tool_metadata[tool_name] = {
                            "load_time": time.time(),
                            "execution_count": 0,
                            "success_count": 0,
                            "error_count": 0,
                            "last_execution": None,
                            "average_execution_time": 0.0
                        }
                        
                        log_debug(f"Loaded tool: {tool_name}", "tools")
                        
                except Exception as e:
                    log_error(f"Error loading tool {tool_file}: {e}", "tools", e)
        
        except Exception as e:
            log_error(f"Error loading tools directory: {e}", "tools", e)
    
    def _validate_tool_definition(self, tool_def: Dict) -> List[str]:
        """Validate tool definition and return list of errors"""
        errors = []
        
        # Required fields
        required_fields = ["name", "description", "function_name", "code"]
        for field in required_fields:
            if not tool_def.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate input parameters
        input_params = tool_def.get("input_parameters", [])
        if not isinstance(input_params, list):
            errors.append("input_parameters must be a list")
        else:
            for i, param in enumerate(input_params):
                if not isinstance(param, dict):
                    errors.append(f"Parameter {i} must be a dictionary")
                    continue
                
                if not param.get("name"):
                    errors.append(f"Parameter {i} missing name")
                if not param.get("type"):
                    errors.append(f"Parameter {i} missing type")
        
        # Validate output parameters
        output_params = tool_def.get("output_parameters", [])
        if not isinstance(output_params, list):
            errors.append("output_parameters must be a list")
        
        return errors

    def _load_tool_function(self, tool_name: str, tool_def: Dict = None):
        """Load the Python function for a tool with enhanced error handling"""
        if tool_def is None:
            log_error(f"No tool definition provided for {tool_name}", "tools")
            return
        
        try:
            code_file = tool_def.get("code")
            if not code_file:
                log_warning(f"No code file specified for tool {tool_name}", "tools")
                return
            
            # Construct path to code file
            code_path = self.tools_directory / "code" / f"{code_file}"
            
            if not code_path.exists():
                log_error(f"Code file not found: {code_path}", "tools")
                return
            
            # Read the code file
            with open(code_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Create a module-like object to execute the code
            namespace = {}
            exec(code_content, namespace)
            
            # Get the function
            function_name = tool_def.get("function_name", tool_name)
            if function_name in namespace:
                func = namespace[function_name]
                
                # Validate function signature
                try:
                    sig = inspect.signature(func)
                    self.tool_functions[tool_name] = func
                    log_debug(f"Loaded function {function_name} for tool {tool_name}", "tools")
                except Exception as e:
                    log_error(f"Invalid function signature for {function_name}: {e}", "tools", e)
            else:
                log_error(f"Function {function_name} not found in code file for tool {tool_name}", "tools")
            
        except Exception as e:
            log_error(f"Error loading function for tool {tool_name}: {e}", "tools", e)
    
    def get_tool_function(self, tool_name: str) -> Optional[Callable]:
        """Get the Python function for a tool"""
        return self.tool_functions.get(tool_name)
    
    def get_tool_definition(self, tool_name: str) -> Optional[Dict]:
        """Get the tool definition"""
        return self.tools_cache.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, Dict]:
        """Get all available tools"""
        return self.tools_cache.copy()
    
    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tool names"""
        enabled_tools = []
        for tool_name, tool_def in self.tools_cache.items():
            if tool_def.get("status") == "enabled":
                enabled_tools.append(tool_name)
        return enabled_tools
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools by category"""
        tools = []
        for tool_name, tool_def in self.tools_cache.items():
            if tool_def.get("category") == category and tool_def.get("status") == "enabled":
                tools.append(tool_name)
        return tools
    
    def create_langchain_tool(self, tool_name: str) -> Optional[Tool]:
        """Create a LangChain Tool object from DurgasAI tool definition"""
        tool_def = self.get_tool_definition(tool_name)
        tool_function = self.get_tool_function(tool_name)
        
        if not tool_def or not tool_function:
            return None
        
        try:
            # Convert DurgasAI tool definition to LangChain format
            description = tool_def.get("description", f"Tool: {tool_name}")
            
            # Parse parameters for LangChain
            parameters = tool_def.get("parameters", {})
            input_parameters = tool_def.get("input_parameters", [])
            
            # Create parameter schema for LangChain
            from pydantic import BaseModel, Field
            from typing import get_type_hints
            
            # Create dynamic Pydantic model for parameters
            if input_parameters:
                # Use create_model for proper Pydantic model creation
                from pydantic import create_model
                
                field_definitions = {}
                for param in input_parameters:
                    param_name = param.get("name")
                    param_type = param.get("type", "string")
                    param_desc = param.get("description", "")
                    required = param.get("required", False)
                    
                    # Convert type string to Python type
                    if param_type == "string":
                        field_type = str
                    elif param_type == "integer":
                        field_type = int
                    elif param_type == "number":
                        field_type = float
                    elif param_type == "boolean":
                        field_type = bool
                    elif param_type == "array":
                        field_type = list
                    else:
                        field_type = str
                    
                    # Create field definition
                    if required:
                        field_definitions[param_name] = (field_type, Field(description=param_desc))
                    else:
                        field_definitions[param_name] = (field_type, Field(default=None, description=param_desc))
                
                # Create the parameter model
                ParameterModel = create_model(f"_{tool_name}_params", **field_definitions)
            else:
                ParameterModel = BaseModel
            
            # Create LangChain Tool
            if input_parameters:
                # Use StructuredTool for tools with parameters
                langchain_tool = StructuredTool.from_function(
                    func=tool_function,
                    name=tool_name,
                    description=description,
                    args_schema=ParameterModel
                )
            else:
                # Use regular Tool for tools without parameters
                langchain_tool = Tool(
                    name=tool_name,
                    description=description,
                    func=tool_function
                )
            
            return langchain_tool
            
        except Exception as e:
            print(f"Error creating LangChain tool for {tool_name}: {e}")
            return None
    
    def get_langchain_tools(self, tool_names: List[str] = None) -> List[Tool]:
        """Get LangChain Tool objects for specified tools"""
        if tool_names is None:
            tool_names = self.get_enabled_tools()
        
        tools = []
        for tool_name in tool_names:
            langchain_tool = self.create_langchain_tool(tool_name)
            if langchain_tool:
                tools.append(langchain_tool)
        
        return tools
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolExecutionResult:
        """Execute a tool with comprehensive error handling and monitoring"""
        start_time = time.time()
        
        try:
            # Validate tool exists
            tool_function = self.get_tool_function(tool_name)
            if not tool_function:
                raise ToolValidationError(f"Tool '{tool_name}' not found or not loaded")
            
            # Validate input parameters with detailed error reporting
            try:
                if not self.validate_tool_input(tool_name, **kwargs):
                    raise ToolValidationError(f"Invalid input parameters for tool '{tool_name}'")
            except Exception as validation_error:
                execution_time = time.time() - start_time
                log_error(f"Tool validation error for '{tool_name}': {str(validation_error)}", "tools", validation_error)
                
                # Update metadata
                self._update_tool_metadata(tool_name, False, execution_time)
                
                return ToolExecutionResult.create_error(
                    error=f"Validation error: {str(validation_error)}",
                    execution_time=execution_time,
                    tool_name=tool_name
                )
            
            # Execute the tool with error handling
            try:
                result = tool_function(**kwargs)
            except Exception as execution_error:
                execution_time = time.time() - start_time
                log_error(f"Tool execution error for '{tool_name}': {str(execution_error)}", "tools", execution_error)
                
                # Update metadata
                self._update_tool_metadata(tool_name, False, execution_time)
                
                return ToolExecutionResult.create_error(
                    error=f"Execution error: {str(execution_error)}",
                    execution_time=execution_time,
                    tool_name=tool_name
                )
            
            execution_time = time.time() - start_time
            
            # Validate result
            if result is None:
                log_warning(f"Tool '{tool_name}' returned None result", "tools")
                result = "Tool executed but returned no result"
            
            # Format result according to output_parameters
            formatted_result = self._format_tool_result(tool_name, result)
            
            # Update metadata
            self._update_tool_metadata(tool_name, True, execution_time)
            
            # Create execution result
            execution_result = ToolExecutionResult.create_success(
                result=result,
                execution_time=execution_time,
                tool_name=tool_name,
                formatted_result=formatted_result
            )
            
            # Store in history with size limit
            self.execution_history.append(execution_result)
            
            # Keep only last 1000 executions to prevent memory issues
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            log_debug(f"Tool '{tool_name}' executed successfully in {execution_time:.3f}s", "tools")
            return execution_result
            
        except ToolValidationError as e:
            execution_time = time.time() - start_time
            log_error(f"Tool validation error for '{tool_name}': {str(e)}", "tools", e)
            
            # Update metadata
            self._update_tool_metadata(tool_name, False, execution_time)
            
            return ToolExecutionResult.create_error(
                error=f"Validation error: {str(e)}",
                execution_time=execution_time,
                tool_name=tool_name
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_error(f"Unexpected error executing tool '{tool_name}': {str(e)}", "tools", e)
            
            # Update metadata
            self._update_tool_metadata(tool_name, False, execution_time)
            
            return ToolExecutionResult.create_error(
                error=f"Unexpected error: {str(e)}",
                execution_time=execution_time,
                tool_name=tool_name
            )

    def _update_tool_metadata(self, tool_name: str, success: bool = True, execution_time: float = 0.0):
        """Update tool metadata after execution"""
        if tool_name in self.tool_metadata:
            metadata = self.tool_metadata[tool_name]
            metadata["execution_count"] += 1
            metadata["last_execution"] = time.time()
            
            if success:
                metadata["success_count"] += 1
            else:
                metadata["error_count"] += 1
            
            # Update average execution time
            total_time = metadata["average_execution_time"] * (metadata["execution_count"] - 1)
            metadata["average_execution_time"] = (total_time + execution_time) / metadata["execution_count"]

    def test_tool(self, tool_name: str, test_params: Dict = None) -> Dict[str, Any]:
        """Test a tool with provided parameters or default test values"""
        try:
            tool_def = self.get_tool_definition(tool_name)
            if not tool_def:
                return {"success": False, "error": f"Tool {tool_name} not found"}
            
            # Generate test parameters if not provided
            if test_params is None:
                test_params = self._generate_test_parameters(tool_name)
            
            # Execute tool
            result = self.execute_tool(tool_name, **test_params)
            
            return {
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "execution_time": result.execution_time,
                "test_parameters": test_params
            }
            
        except Exception as e:
            log_error(f"Error testing tool {tool_name}: {e}", "tools", e)
            return {"success": False, "error": str(e)}

    def _generate_test_parameters(self, tool_name: str) -> Dict[str, Any]:
        """Generate test parameters for a tool based on its definition"""
        tool_def = self.get_tool_definition(tool_name)
        if not tool_def:
            return {}
        
        test_params = {}
        input_params = tool_def.get("input_parameters", [])
        
        for param in input_params:
            param_name = param.get("name")
            param_type = param.get("type", "string")
            required = param.get("required", False)
            
            if not param_name:
                continue
            
            # Generate test value based on type
            if param_type == "string":
                test_params[param_name] = f"test_{param_name}"
            elif param_type == "integer":
                test_params[param_name] = 42
            elif param_type == "number":
                test_params[param_name] = 3.14
            elif param_type == "boolean":
                test_params[param_name] = True
            elif param_type == "array":
                test_params[param_name] = ["item1", "item2"]
            else:
                test_params[param_name] = f"test_{param_name}"
        
        return test_params
    
    def validate_tool_input(self, tool_name: str, **kwargs) -> bool:
        """Enhanced tool input validation with type checking"""
        tool_def = self.get_tool_definition(tool_name)
        if not tool_def:
            return False
        
        input_parameters = tool_def.get("input_parameters", [])
        required_params = [p["name"] for p in input_parameters if p.get("required", False)]
        
        # Check required parameters
        for param in required_params:
            if param not in kwargs:
                log_debug(f"Missing required parameter {param} for tool {tool_name}", "tools")
                return False
        
        # Type validation
        for param in input_parameters:
            param_name = param.get("name")
            param_type = param.get("type", "string")
            
            if param_name in kwargs:
                value = kwargs[param_name]
                if not self._validate_parameter_type(value, param_type):
                    log_debug(f"Invalid type for parameter {param_name} in tool {tool_name}", "tools")
                    return False
        
        return True

    def _validate_parameter_type(self, value: Any, expected_type: str) -> bool:
        """Validate parameter type"""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        else:
            return True  # Unknown type, assume valid

    def _format_tool_result(self, tool_name: str, result: Any) -> Dict[str, Any]:
        """Format tool result according to output_parameters definition"""
        tool_def = self.get_tool_definition(tool_name)
        if not tool_def:
            return {"raw_result": result}
        
        output_parameters = tool_def.get("output_parameters", [])
        if not output_parameters:
            return {"raw_result": result}
        
        formatted_result = {}
        
        # Try to parse result as JSON if it's a string
        parsed_result = result
        if isinstance(result, str):
            try:
                import json
                parsed_result = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, keep as string
                parsed_result = result
        
        # If result is a dictionary, try to map it to output_parameters
        if isinstance(parsed_result, dict):
            for param in output_parameters:
                param_name = param.get("name")
                param_type = param.get("type", "string")
                param_description = param.get("description", "")
                param_format = param.get("format", "plain_text")
                
                if param_name in parsed_result:
                    formatted_result[param_name] = {
                        "value": parsed_result[param_name],
                        "type": param_type,
                        "description": param_description,
                        "format": param_format
                    }
                else:
                    # If parameter not found in result, show as missing
                    formatted_result[param_name] = {
                        "value": None,
                        "type": param_type,
                        "description": param_description,
                        "format": param_format,
                        "status": "missing"
                    }
        else:
            # If result is not a dictionary, try to map to first output parameter
            if output_parameters:
                first_param = output_parameters[0]
                formatted_result[first_param.get("name", "result")] = {
                    "value": parsed_result,
                    "type": first_param.get("type", "string"),
                    "description": first_param.get("description", ""),
                    "format": first_param.get("format", "plain_text")
                }
        
        return formatted_result

    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tool statistics"""
        total_tools = len(self.tools_cache)
        enabled_tools = len(self.get_enabled_tools())
        
        # Calculate execution statistics
        total_executions = sum(meta["execution_count"] for meta in self.tool_metadata.values())
        total_successes = sum(meta["success_count"] for meta in self.tool_metadata.values())
        total_errors = sum(meta["error_count"] for meta in self.tool_metadata.values())
        
        # Most used tools
        most_used = sorted(
            [(name, meta["execution_count"]) for name, meta in self.tool_metadata.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Tool categories
        categories = {}
        for tool_def in self.tools_cache.values():
            category = tool_def.get("category", "uncategorized")
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "disabled_tools": total_tools - enabled_tools,
            "total_executions": total_executions,
            "success_rate": (total_successes / total_executions * 100) if total_executions > 0 else 0,
            "error_rate": (total_errors / total_executions * 100) if total_executions > 0 else 0,
            "most_used_tools": most_used,
            "tool_categories": categories,
            "recent_executions": len([r for r in self.execution_history if time.time() - r.timestamp < 3600])  # Last hour
        }

    def get_tool_performance_report(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed performance report for a specific tool"""
        if tool_name not in self.tool_metadata:
            return {"error": f"Tool {tool_name} not found"}
        
        metadata = self.tool_metadata[tool_name]
        tool_def = self.get_tool_definition(tool_name)
        
        # Recent executions
        recent_executions = [
            r for r in self.execution_history 
            if r.tool_name == tool_name and time.time() - r.timestamp < 3600
        ]
        
        return {
            "tool_name": tool_name,
            "description": tool_def.get("description", ""),
            "category": tool_def.get("category", ""),
            "status": tool_def.get("status", ""),
            "execution_count": metadata["execution_count"],
            "success_count": metadata["success_count"],
            "error_count": metadata["error_count"],
            "success_rate": (metadata["success_count"] / metadata["execution_count"] * 100) if metadata["execution_count"] > 0 else 0,
            "average_execution_time": metadata["average_execution_time"],
            "last_execution": metadata["last_execution"],
            "recent_executions_count": len(recent_executions),
            "load_time": metadata["load_time"]
        }

    def batch_test_tools(self, tool_names: List[str] = None) -> Dict[str, Any]:
        """Test multiple tools in batch"""
        if tool_names is None:
            tool_names = self.get_enabled_tools()
        
        results = {}
        successful_tests = 0
        failed_tests = 0
        
        for tool_name in tool_names:
            test_result = self.test_tool(tool_name)
            results[tool_name] = test_result
            
            if test_result["success"]:
                successful_tests += 1
            else:
                failed_tests += 1
        
        return {
            "total_tools_tested": len(tool_names),
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests / len(tool_names) * 100) if tool_names else 0,
            "results": results
        }
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get comprehensive tool information with performance data"""
        tool_def = self.get_tool_definition(tool_name)
        if not tool_def:
            return {}
        
        metadata = self.tool_metadata.get(tool_name, {})
        
        return {
            "name": tool_def.get("name"),
            "description": tool_def.get("description"),
            "category": tool_def.get("category"),
            "status": tool_def.get("status"),
            "version": tool_def.get("version"),
            "author": tool_def.get("author"),
            "parameters": tool_def.get("input_parameters", []),
            "output_parameters": tool_def.get("output_parameters", []),
            "tags": tool_def.get("tags", []),
            "security_level": tool_def.get("security_level"),
            "has_function": tool_name in self.tool_functions,
            "created_at": tool_def.get("created_at"),
            "updated_at": tool_def.get("updated_at"),
            "performance": {
                "execution_count": metadata.get("execution_count", 0),
                "success_count": metadata.get("success_count", 0),
                "error_count": metadata.get("error_count", 0),
                "success_rate": (metadata.get("success_count", 0) / metadata.get("execution_count", 1) * 100) if metadata.get("execution_count", 0) > 0 else 0,
                "average_execution_time": metadata.get("average_execution_time", 0.0),
                "last_execution": metadata.get("last_execution")
            }
        }
    
    def reload_tools(self):
        """Reload all tools from disk with enhanced logging"""
        log_info("Reloading tools from disk", "tools")
        self.tools_cache.clear()
        self.tool_functions.clear()
        self.tool_metadata.clear()
        self._load_tools()
        log_info(f"Reloaded {len(self.tools_cache)} tools", "tools")
    
    def add_tool(self, tool_name: str, tool_def: Dict = None, tool_function: Callable = None):
        """Add a new tool dynamically with validation"""
        if tool_def is None:
            log_error(f"No tool definition provided for {tool_name}", "tools")
            return False
        
        if tool_function is None:
            log_error(f"No tool function provided for {tool_name}", "tools")
            return False
        
        try:
            # Validate tool definition
            validation_errors = self._validate_tool_definition(tool_def)
            if validation_errors:
                log_warning(f"Tool {tool_name} has validation issues: {validation_errors}", "tools")
            
            self.tools_cache[tool_name] = tool_def
            self.tool_functions[tool_name] = tool_function
            
            # Initialize metadata
            self.tool_metadata[tool_name] = {
                "load_time": time.time(),
                "execution_count": 0,
                "success_count": 0,
                "error_count": 0,
                "last_execution": None,
                "average_execution_time": 0.0
            }
            
            log_info(f"Added tool: {tool_name}", "tools")
            
        except Exception as e:
            log_error(f"Error adding tool {tool_name}: {e}", "tools", e)
            raise
    
    def remove_tool(self, tool_name: str):
        """Remove a tool with cleanup"""
        try:
            if tool_name in self.tools_cache:
                del self.tools_cache[tool_name]
                log_debug(f"Removed tool definition: {tool_name}", "tools")
            
            if tool_name in self.tool_functions:
                del self.tool_functions[tool_name]
                log_debug(f"Removed tool function: {tool_name}", "tools")
            
            if tool_name in self.tool_metadata:
                del self.tool_metadata[tool_name]
                log_debug(f"Removed tool metadata: {tool_name}", "tools")
            
            log_info(f"Removed tool: {tool_name}", "tools")
            
        except Exception as e:
            log_error(f"Error removing tool {tool_name}: {e}", "tools", e)
            raise
