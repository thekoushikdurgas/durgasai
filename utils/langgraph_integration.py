"""
LangGraph Integration Manager for DurgasAI
Integrates LangGraph chatbot functionality with DurgasAI tools and templates
Enhanced with advanced workflow patterns and better state management
"""

import json
import os
import uuid
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from utils.tool_manager import DurgasAIToolManager
from utils.config_manager import ConfigManager
from utils.logging_utils import log_info, log_error, log_debug, log_warning

class ChatState(TypedDict):
    """Enhanced state schema for LangGraph workflows"""
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    workflow_data: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    tool_responses: List[Dict[str, Any]]
    error_count: int
    max_iterations: int

class LangGraphManager:
    """Enhanced LangGraph chatbot integration with DurgasAI"""
    
    def __init__(self, config_manager: ConfigManager, tool_manager: DurgasAIToolManager):
        self.config_manager = config_manager
        self.tool_manager = tool_manager
        self.active_sessions = {}
        self.workflow_functions = {}
        self._initialize_workflow_functions()
    
    def _initialize_workflow_functions(self):
        """Initialize workflow creation functions"""
        self.workflow_functions = {
            "basic_chat": self._create_basic_chat_workflow,
            "advanced_tools": self._create_advanced_tools_workflow,
            "research_assistant": self._create_research_assistant_workflow,
            "code_assistant": self._create_code_assistant_workflow,
            "conditional_workflow": self._create_conditional_workflow
        }
        log_info("Workflow functions initialized", "langgraph")

    def create_chat_session(self, template: Dict, llm_provider: str, model: str, api_keys: Dict, workflow_type: str = "basic_chat") -> str:
        """Create a new chat session with enhanced LangGraph workflow"""
        session_id = str(uuid.uuid4())
        
        try:
            log_info(f"Creating chat session {session_id} with workflow {workflow_type}", "langgraph")
            
            # Initialize LLM
            llm = self._initialize_llm(llm_provider, model, api_keys)
            
            # Get tools for the template
            tools = self._get_tools_for_template(template)
            
            # Create enhanced LangGraph workflow
            graph = self._create_enhanced_workflow(llm, tools, template, workflow_type)
            
            # Store session with enhanced metadata
            self.active_sessions[session_id] = {
                "graph": graph,
                "llm": llm,
                "tools": tools,
                "template": template,
                "workflow_type": workflow_type,
                "history": [],
                "created_at": self._get_timestamp(),
                "last_activity": self._get_timestamp(),
                "message_count": 0,
                "tool_calls_count": 0
            }
            
            log_info(f"Chat session {session_id} created successfully", "langgraph")
            return session_id
            
        except Exception as e:
            log_error(f"Failed to create chat session: {str(e)}", "langgraph", e)
            raise Exception(f"Failed to create chat session: {str(e)}")
    
    def process_message(self, session: str, message: str, template: Dict) -> Dict:
        """Process a message through the enhanced LangGraph workflow with comprehensive error handling"""
        start_time = time.time()
        
        # Validate session
        if session not in self.active_sessions:
            log_error(f"Invalid session ID: {session}", "langgraph")
            return {
                "content": "I'm sorry, but I couldn't find your session. Please start a new conversation.",
                "error": "session_not_found",
                "tool_calls": [],
                "workflow_data": {"error_type": "session_not_found"},
                "tool_responses": [],
                "session_metadata": {}
            }
        
        # Validate input
        if not message or not isinstance(message, str) or not message.strip():
            log_warning(f"Invalid message input for session {session}", "langgraph")
            return {
                "content": "I received an invalid message. Please try again with a proper text message.",
                "error": "invalid_input",
                "tool_calls": [],
                "workflow_data": {"error_type": "invalid_input"},
                "tool_responses": [],
                "session_metadata": {}
            }
        
        session_data = self.active_sessions[session]
        graph = session_data["graph"]
        
        try:
            log_debug(f"Processing message in session {session}", "langgraph")
            
            # Create enhanced initial state with validation
            initial_state = {
                "messages": [HumanMessage(content=message.strip())],
                "current_step": "chat",
                "workflow_data": {
                    "session_id": session,
                    "template_name": template.get("name", "unknown"),
                    "processing_start": datetime.now().isoformat(),
                    "message_length": len(message)
                },
                "tool_calls": [],
                "tool_responses": [],
                "error_count": 0,
                "max_iterations": 10
            }
            
            # Process through graph with timeout protection
            try:
                result = graph.invoke(initial_state)
            except Exception as graph_error:
                log_error(f"Graph execution failed for session {session}: {str(graph_error)}", "langgraph", graph_error)
                return {
                    "content": "I encountered an error while processing your request. Please try again.",
                    "error": "graph_execution_failed",
                    "tool_calls": [],
                    "workflow_data": {
                        "error_type": "graph_execution_failed",
                        "error_details": str(graph_error),
                        "processing_time": time.time() - start_time
                    },
                    "tool_responses": [],
                    "session_metadata": {}
                }
            
            # Extract enhanced response data with validation
            messages = result.get("messages", [])
            workflow_data = result.get("workflow_data", {})
            tool_calls = result.get("tool_calls", [])
            tool_responses = result.get("tool_responses", [])
            
            # Find the last assistant message with error handling
            assistant_response = None
            try:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        assistant_response = msg.content
                        break
            except Exception as msg_error:
                log_error(f"Error extracting assistant response: {str(msg_error)}", "langgraph", msg_error)
                assistant_response = "I encountered an error while generating my response."
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update session metadata with error handling
            try:
                session_data["last_activity"] = self._get_timestamp()
                session_data["message_count"] = session_data.get("message_count", 0) + 1
                session_data["tool_calls_count"] = session_data.get("tool_calls_count", 0) + len(tool_calls)
                session_data["total_processing_time"] = session_data.get("total_processing_time", 0) + processing_time
                
                # Add to history with size limit
                if "history" not in session_data:
                    session_data["history"] = []
                session_data["history"].extend(messages)
                
                # Keep only last 100 messages to prevent memory issues
                if len(session_data["history"]) > 100:
                    session_data["history"] = session_data["history"][-100:]
                    
            except Exception as metadata_error:
                log_error(f"Error updating session metadata: {str(metadata_error)}", "langgraph", metadata_error)
            
            # Enhanced workflow data
            enhanced_workflow_data = {
                **workflow_data,
                "processing_time": processing_time,
                "error_count": result.get("error_count", 0),
                "message_count": session_data.get("message_count", 0),
                "tokens_estimated": self._estimate_tokens(assistant_response) if assistant_response else 0
            }
            
            log_info(f"Message processed successfully in session {session} in {processing_time:.2f}s", "langgraph")
            
            return {
                "content": assistant_response or "I'm sorry, I couldn't generate a response.",
                "tool_calls": tool_calls,
                "tool_responses": tool_responses,
                "workflow_data": enhanced_workflow_data,
                "session_metadata": {
                    "message_count": session_data.get("message_count", 0),
                    "tool_calls_count": session_data.get("tool_calls_count", 0),
                    "last_activity": session_data.get("last_activity", ""),
                    "total_processing_time": session_data.get("total_processing_time", 0)
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            log_error(f"Critical error processing message in session {session}: {str(e)}", "langgraph", e)
            
            # Update error count in session
            try:
                session_data["error_count"] = session_data.get("error_count", 0) + 1
                session_data["last_error"] = str(e)
                session_data["last_error_time"] = self._get_timestamp()
            except Exception as error_update_error:
                log_error(f"Error updating error metadata: {str(error_update_error)}", "langgraph", error_update_error)
            
            return {
                "content": "I encountered a critical error while processing your request. Please try again or contact support if the issue persists.",
                "error": "critical_error",
                "tool_calls": [],
                "workflow_data": {
                    "error_type": "critical_error",
                    "error_details": str(e),
                    "processing_time": processing_time
                },
                "tool_responses": [],
                "session_metadata": {
                    "error_count": session_data.get("error_count", 0),
                    "last_error": str(e)
                }
            }
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)"""
        if not text:
            return 0
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _initialize_llm(self, provider: str = "OpenAI", model: str = "gpt-4o", api_keys: Dict = None):
        """Initialize the LLM based on provider and model with enhanced configuration"""
        if api_keys is None:
            api_keys = {}
        
        try:
            if provider.lower() == "openai":
                api_key = api_keys.get("openai")
                if not api_key:
                    raise Exception("OpenAI API key not provided")
                
                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30
                )
            
            elif provider.lower() == "groq":
                api_key = api_keys.get("groq")
                if not api_key:
                    raise Exception("Groq API key not provided")
                
                return ChatGroq(
                    model=model,
                    api_key=api_key,
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30
                )
            
            elif provider.lower() == "anthropic":
                api_key = api_keys.get("anthropic")
                if not api_key:
                    raise Exception("Anthropic API key not provided")
                
                return ChatAnthropic(
                    model=model,
                    api_key=api_key,
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30
                )
            
            elif provider.lower() == "google":
                api_key = api_keys.get("google")
                if not api_key:
                    raise Exception("Google API key not provided")
                
                return ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=0.7,
                    max_output_tokens=2048,
                    timeout=30
                )
            
            elif provider.lower() == "github":
                api_key = api_keys.get("github")
                if not api_key:
                    raise Exception("GitHub API key not provided")
                
                # GitHub Models uses OpenAI-compatible API
                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url="https://models.inference.ai.azure.com",
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30
                )
            
            elif provider.lower() == "openrouter":
                api_key = api_keys.get("openrouter")
                if not api_key:
                    raise Exception("OpenRouter API key not provided")
                
                # OpenRouter uses OpenAI-compatible API
                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30,
                    default_headers={
                        "HTTP-Referer": "https://github.com/DurgasAI/DurgasAI",
                        "X-Title": "DurgasAI"
                    }
                )
            
            elif provider.lower() == "nvidia":
                api_key = api_keys.get("nvidia")
                if not api_key:
                    raise Exception("NVIDIA API key not provided")
                
                # NVIDIA uses OpenAI-compatible API
                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url="https://integrate.api.nvidia.com/v1",
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30
                )
            
            elif provider.lower() == "cohere":
                api_key = api_keys.get("cohere")
                if not api_key:
                    raise Exception("Cohere API key not provided")
                
                return ChatCohere(
                    model=model,
                    cohere_api_key=api_key,
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30
                )
            
            else:
                raise Exception(f"Unsupported LLM provider: {provider}")
                
        except Exception as e:
            log_error(f"Failed to initialize LLM {provider}: {str(e)}", "langgraph", e)
            raise
    
    def _get_tools_for_template(self, template: Dict) -> List:
        """Get tools for the given template"""
        enabled_tools = template.get("function_calling_tools", [])
        
        if not enabled_tools:
            return []
        
        # Get LangChain tools from DurgasAI tool manager
        tools = self.tool_manager.get_langchain_tools(enabled_tools)
        
        return tools
    
    def _create_enhanced_workflow(self, llm = None, tools: List = None, template: Dict = None, workflow_type: str = "basic_chat"):
        """Create enhanced LangGraph workflow with advanced patterns"""
        if llm is None:
            raise ValueError("LLM instance is required")
        if tools is None:
            tools = []
        if template is None:
            template = {}
        
        try:
            # Get workflow creation function
            if workflow_type in self.workflow_functions:
                return self.workflow_functions[workflow_type](llm, tools, template)
            else:
                log_debug(f"Unknown workflow type {workflow_type}, using basic_chat", "langgraph")
                return self.workflow_functions["basic_chat"](llm, tools, template)
        except Exception as e:
            log_error(f"Failed to create workflow {workflow_type}: {str(e)}", "langgraph", e)
            raise

    def _create_basic_chat_workflow(self, llm = None, tools: List = None, template: Dict = None):
        """Create a basic chat workflow"""
        if llm is None:
            raise ValueError("LLM instance is required")
        if tools is None:
            tools = []
        if template is None:
            template = {}
        
        workflow = StateGraph(ChatState)
        
        # Bind tools to LLM
        if tools:
            llm_with_tools = llm.bind_tools(tools)
        else:
            llm_with_tools = llm
        
        def chatbot_node(state: ChatState):
            """Enhanced chatbot node with better error handling"""
            try:
                system_prompt = template.get("system_instruction", "You are a helpful AI assistant.")
                
                messages = state["messages"]
                # Add system message if not present
                if messages and not isinstance(messages[0], SystemMessage):
                    messages = [SystemMessage(content=system_prompt)] + messages
                
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                log_error(f"Error in chatbot node: {str(e)}", "langgraph", e)
                error_msg = AIMessage(content=f"I encountered an error: {str(e)}")
                return {"messages": [error_msg]}
        
        # Add nodes
        workflow.add_node("chatbot", chatbot_node)
        if tools:
            tool_node = ToolNode(tools)
            workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.add_edge(START, "chatbot")
        if tools:
            workflow.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
            workflow.add_edge("tools", "chatbot")
        else:
            workflow.add_edge("chatbot", END)
        
        return workflow.compile()

    def _create_advanced_tools_workflow(self, llm, tools: List, template: Dict):
        """Create an advanced workflow with sophisticated tool usage"""
        workflow = StateGraph(ChatState)
        
        # Bind tools to LLM
        if tools:
            llm_with_tools = llm.bind_tools(tools)
        else:
            llm_with_tools = llm
        
        def chatbot_node(state: ChatState):
            """Advanced chatbot node with tool planning"""
            try:
                system_prompt = template.get("system_instruction", "You are a helpful AI assistant.")
                system_prompt += "\n\nWhen using tools, think step by step and explain your reasoning."
                
                messages = state["messages"]
                if messages and not isinstance(messages[0], SystemMessage):
                    messages = [SystemMessage(content=system_prompt)] + messages
                
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                log_error(f"Error in advanced chatbot node: {str(e)}", "langgraph", e)
                error_msg = AIMessage(content=f"I encountered an error: {str(e)}")
                return {"messages": [error_msg]}
        
        def tool_planner_node(state: ChatState):
            """Node that plans tool usage"""
            # This could be enhanced with more sophisticated planning logic
            return state
        
        # Add nodes
        workflow.add_node("chatbot", chatbot_node)
        workflow.add_node("planner", tool_planner_node)
        if tools:
            tool_node = ToolNode(tools)
            workflow.add_node("tools", tool_node)
        
        # Add edges with conditional routing
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "chatbot")
        if tools:
            workflow.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
            workflow.add_edge("tools", "chatbot")
        else:
            workflow.add_edge("chatbot", END)
        
        return workflow.compile()

    def _create_research_assistant_workflow(self, llm, tools: List, template: Dict):
        """Create a research assistant workflow with web search capabilities"""
        workflow = StateGraph(ChatState)
        
        # Filter tools for research capabilities
        research_tools = [tool for tool in tools if any(keyword in tool.name.lower() 
                        for keyword in ['search', 'web', 'browser', 'tavily'])]
        
        if research_tools:
            llm_with_tools = llm.bind_tools(research_tools)
        else:
            llm_with_tools = llm
        
        def research_node(state: ChatState):
            """Research-focused chatbot node"""
            try:
                system_prompt = template.get("system_instruction", "You are a helpful AI assistant.")
                system_prompt += "\n\nYou are a research assistant. Use web search tools to find current information and provide comprehensive, well-sourced answers."
                
                messages = state["messages"]
                if messages and not isinstance(messages[0], SystemMessage):
                    messages = [SystemMessage(content=system_prompt)] + messages
                
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                log_error(f"Error in research node: {str(e)}", "langgraph", e)
                error_msg = AIMessage(content=f"I encountered an error: {str(e)}")
                return {"messages": [error_msg]}
        
        # Add nodes
        workflow.add_node("researcher", research_node)
        if research_tools:
            tool_node = ToolNode(research_tools)
            workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.add_edge(START, "researcher")
        if research_tools:
            workflow.add_conditional_edges("researcher", tools_condition, {"tools": "tools", END: END})
            workflow.add_edge("tools", "researcher")
        else:
            workflow.add_edge("researcher", END)
        
        return workflow.compile()

    def _create_code_assistant_workflow(self, llm, tools: List, template: Dict):
        """Create a code assistant workflow with development tools"""
        workflow = StateGraph(ChatState)
        
        # Filter tools for code capabilities
        code_tools = [tool for tool in tools if any(keyword in tool.name.lower() 
                      for keyword in ['calculate', 'math', 'file', 'code', 'analyze'])]
        
        if code_tools:
            llm_with_tools = llm.bind_tools(code_tools)
        else:
            llm_with_tools = llm
        
        def code_node(state: ChatState):
            """Code-focused chatbot node"""
            try:
                system_prompt = template.get("system_instruction", "You are a helpful AI assistant.")
                system_prompt += "\n\nYou are a code assistant. Help with programming, calculations, file operations, and code analysis. Use tools when appropriate to demonstrate or verify solutions."
                
                messages = state["messages"]
                if messages and not isinstance(messages[0], SystemMessage):
                    messages = [SystemMessage(content=system_prompt)] + messages
                
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                log_error(f"Error in code node: {str(e)}", "langgraph", e)
                error_msg = AIMessage(content=f"I encountered an error: {str(e)}")
                return {"messages": [error_msg]}
        
        # Add nodes
        workflow.add_node("coder", code_node)
        if code_tools:
            tool_node = ToolNode(code_tools)
            workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.add_edge(START, "coder")
        if code_tools:
            workflow.add_conditional_edges("coder", tools_condition, {"tools": "tools", END: END})
            workflow.add_edge("tools", "coder")
        else:
            workflow.add_edge("coder", END)
        
        return workflow.compile()
    
    def _create_conditional_workflow(self, llm, tools: List, template: Dict):
        """Create a workflow with advanced conditional routing based on input analysis"""
        workflow = StateGraph(ChatState)
        
        def input_analyzer_node(state: ChatState):
            """Analyze input to determine workflow path"""
            try:
                messages = state["messages"]
                last_message = messages[-1] if messages else None
                
                if last_message and isinstance(last_message, HumanMessage):
                    content = last_message.content.lower()
                    
                    # Analyze content for different workflow paths
                    if any(keyword in content for keyword in ["calculate", "math", "compute", "number"]):
                        return {
                            "current_step": "math_processing",
                            "workflow_data": {
                                **state.get("workflow_data", {}),
                                "workflow_type": "math",
                                "requires_tools": True
                            }
                        }
                    elif any(keyword in content for keyword in ["search", "find", "lookup", "web"]):
                        return {
                            "current_step": "search_processing",
                            "workflow_data": {
                                **state.get("workflow_data", {}),
                                "workflow_type": "search",
                                "requires_tools": True
                            }
                        }
                    elif any(keyword in content for keyword in ["code", "program", "script", "function"]):
                        return {
                            "current_step": "code_processing",
                            "workflow_data": {
                                **state.get("workflow_data", {}),
                                "workflow_type": "code",
                                "requires_tools": True
                            }
                        }
                    else:
                        return {
                            "current_step": "general_processing",
                            "workflow_data": {
                                **state.get("workflow_data", {}),
                                "workflow_type": "general",
                                "requires_tools": False
                            }
                        }
                else:
                    return {
                        "current_step": "general_processing",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "workflow_type": "general",
                            "requires_tools": False
                        }
                    }
            except Exception as e:
                log_error(f"Error in input analyzer: {str(e)}", "langgraph", e)
                return {
                    "current_step": "error",
                    "error_count": state.get("error_count", 0) + 1
                }
        
        def math_processing_node(state: ChatState):
            """Process mathematical requests"""
            try:
                # Filter for math-related tools
                math_tools = [tool for tool in tools if any(keyword in tool.name.lower() 
                                 for keyword in ['calculate', 'math', 'add', 'subtract', 'multiply', 'divide'])]
                
                if math_tools:
                    llm_with_tools = llm.bind_tools(math_tools)
                    response = llm_with_tools.invoke(state["messages"])
                    return {
                        "messages": [response],
                        "current_step": "tool_execution",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "specialized_tools": [tool.name for tool in math_tools]
                        }
                    }
                else:
                    # Fallback to general processing
                    response = llm.invoke(state["messages"])
                    return {
                        "messages": [response],
                        "current_step": "response_generation",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "fallback_used": True
                        }
                    }
            except Exception as e:
                log_error(f"Error in math processing: {str(e)}", "langgraph", e)
                return {
                    "current_step": "error",
                    "error_count": state.get("error_count", 0) + 1
                }
        
        def search_processing_node(state: ChatState):
            """Process search requests"""
            try:
                # Filter for search-related tools
                search_tools = [tool for tool in tools if any(keyword in tool.name.lower() 
                                 for keyword in ['search', 'web', 'browser', 'tavily'])]
                
                if search_tools:
                    llm_with_tools = llm.bind_tools(search_tools)
                    response = llm_with_tools.invoke(state["messages"])
                    return {
                        "messages": [response],
                        "current_step": "tool_execution",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "specialized_tools": [tool.name for tool in search_tools]
                        }
                    }
                else:
                    # Fallback to general processing
                    response = llm.invoke(state["messages"])
                    return {
                        "messages": [response],
                        "current_step": "response_generation",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "fallback_used": True
                        }
                    }
            except Exception as e:
                log_error(f"Error in search processing: {str(e)}", "langgraph", e)
                return {
                    "current_step": "error",
                    "error_count": state.get("error_count", 0) + 1
                }
        
        def code_processing_node(state: ChatState):
            """Process code-related requests"""
            try:
                # Filter for code-related tools
                code_tools = [tool for tool in tools if any(keyword in tool.name.lower() 
                                 for keyword in ['code', 'file', 'analyze', 'text'])]
                
                if code_tools:
                    llm_with_tools = llm.bind_tools(code_tools)
                    response = llm_with_tools.invoke(state["messages"])
                    return {
                        "messages": [response],
                        "current_step": "tool_execution",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "specialized_tools": [tool.name for tool in code_tools]
                        }
                    }
                else:
                    # Fallback to general processing
                    response = llm.invoke(state["messages"])
                    return {
                        "messages": [response],
                        "current_step": "response_generation",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "fallback_used": True
                        }
                    }
            except Exception as e:
                log_error(f"Error in code processing: {str(e)}", "langgraph", e)
                return {
                    "current_step": "error",
                    "error_count": state.get("error_count", 0) + 1
                }
        
        def general_processing_node(state: ChatState):
            """Process general requests"""
            try:
                # Use all available tools for general processing
                llm_with_tools = llm.bind_tools(tools)
                response = llm_with_tools.invoke(state["messages"])
                return {
                    "messages": [response],
                    "current_step": "tool_execution",
                    "workflow_data": {
                        **state.get("workflow_data", {}),
                        "general_processing": True
                    }
                }
            except Exception as e:
                log_error(f"Error in general processing: {str(e)}", "langgraph", e)
                return {
                    "current_step": "error",
                    "error_count": state.get("error_count", 0) + 1
                }
        
        def tools_node(state: ChatState):
            """Execute tools based on workflow type"""
            try:
                messages = state["messages"]
                last_message = messages[-1] if messages else None
                
                if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    # Execute tool calls using LangGraph's ToolNode
                    tool_messages = []
                    for tool_call in last_message.tool_calls:
                        try:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            
                            # Find the corresponding LangChain tool
                            tool_obj = None
                            for tool in tools:
                                if tool.name == tool_name:
                                    tool_obj = tool
                                    break
                            
                            if tool_obj:
                                # Execute using LangChain tool
                                result = tool_obj.invoke(tool_args)
                                tool_message = ToolMessage(
                                    content=str(result),
                                    tool_call_id=tool_call["id"]
                                )
                            else:
                                # Fallback to DurgasAI tool manager
                                result = self.tool_manager.execute_tool(tool_name, **tool_args)
                                tool_message = ToolMessage(
                                    content=str(result.result) if result.success else f"Error: {result.error}",
                                    tool_call_id=tool_call["id"]
                                )
                            
                            tool_messages.append(tool_message)
                            
                        except Exception as e:
                            log_error(f"Error executing tool {tool_call['name']}: {str(e)}", "langgraph", e)
                            tool_messages.append(ToolMessage(
                                content=f"Error executing tool: {str(e)}",
                                tool_call_id=tool_call["id"]
                            ))
                    
                    return {
                        "messages": tool_messages,
                        "current_step": "response_generation",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "tools_executed": len(tool_messages)
                        }
                    }
                else:
                    return {
                        "current_step": "response_generation",
                        "workflow_data": {
                            **state.get("workflow_data", {}),
                            "no_tools_executed": True
                        }
                    }
            except Exception as e:
                log_error(f"Error in tools node: {str(e)}", "langgraph", e)
                return {
                    "current_step": "error",
                    "error_count": state.get("error_count", 0) + 1
                }
        
        def conditional_router(state: ChatState):
            """Advanced conditional routing based on workflow state"""
            current_step = state.get("current_step", "input_analyzer")
            error_count = state.get("error_count", 0)
            max_iterations = state.get("max_iterations", 10)
            
            # Check for error conditions
            if error_count >= 3:
                return "end"
            
            # Check iteration limit
            if len(state.get("messages", [])) > max_iterations:
                return "end"
            
            # Route based on current step
            if current_step == "input_analyzer":
                return "input_analyzer"
            elif current_step == "math_processing":
                return "math_processing"
            elif current_step == "search_processing":
                return "search_processing"
            elif current_step == "code_processing":
                return "code_processing"
            elif current_step == "general_processing":
                return "general_processing"
            elif current_step == "tool_execution":
                return "tools"
            elif current_step == "response_generation":
                return "end"
            elif current_step == "error":
                return "end"
            else:
                return "end"
        
        # Add nodes
        workflow.add_node("input_analyzer", input_analyzer_node)
        workflow.add_node("math_processing", math_processing_node)
        workflow.add_node("search_processing", search_processing_node)
        workflow.add_node("code_processing", code_processing_node)
        workflow.add_node("general_processing", general_processing_node)
        workflow.add_node("tools", tools_node)
        
        # Add edges
        workflow.add_edge(START, "input_analyzer")
        workflow.add_conditional_edges("input_analyzer", conditional_router)
        workflow.add_conditional_edges("math_processing", conditional_router)
        workflow.add_conditional_edges("search_processing", conditional_router)
        workflow.add_conditional_edges("code_processing", conditional_router)
        workflow.add_conditional_edges("general_processing", conditional_router)
        workflow.add_conditional_edges("tools", conditional_router)
        
        return workflow.compile()
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get enhanced information about a chat session"""
        session_data = self.active_sessions.get(session_id)
        if session_data:
            return {
                "session_id": session_id,
                "workflow_type": session_data.get("workflow_type"),
                "template": session_data.get("template", {}).get("name"),
                "created_at": session_data.get("created_at"),
                "last_activity": session_data.get("last_activity"),
                "message_count": session_data.get("message_count", 0),
                "tool_calls_count": session_data.get("tool_calls_count", 0),
                "tools_available": len(session_data.get("tools", [])),
                "status": "active"
            }
        return None
    
    def get_all_sessions(self) -> List[Dict]:
        """Get information about all active sessions"""
        return [self.get_session_info(session_id) for session_id in self.active_sessions.keys()]
    
    def cleanup_session(self, session_id: str = None):
        """Clean up a chat session with logging"""
        if session_id is None:
            return
        
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            log_info(f"Cleaning up session {session_id} with {session_data.get('message_count', 0)} messages", "langgraph")
            del self.active_sessions[session_id]
        else:
            log_debug(f"Session {session_id} not found for cleanup", "langgraph")
    
    def cleanup_all_sessions(self):
        """Clean up all chat sessions with logging"""
        session_count = len(self.active_sessions)
        log_info(f"Cleaning up {session_count} active sessions", "langgraph")
        self.active_sessions.clear()
    
    def get_workflow_templates(self) -> List[str]:
        """Get available workflow function types"""
        return list(self.workflow_functions.keys())
    
    def add_custom_workflow(self, name: str = None, workflow_func = None):
        """Add a custom workflow template"""
        if name is None or workflow_func is None:
            log_error("Both name and workflow_func are required for custom workflow", "langgraph")
            return
        
        self.workflow_functions[name] = workflow_func
        log_info(f"Added custom workflow function: {name}", "langgraph")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about all sessions"""
        sessions = list(self.active_sessions.values())
        if not sessions:
            return {"total_sessions": 0}
        
        total_messages = sum(s.get("message_count", 0) for s in sessions)
        total_tool_calls = sum(s.get("tool_calls_count", 0) for s in sessions)
        workflow_types = {}
        
        for session in sessions:
            workflow_type = session.get("workflow_type", "unknown")
            workflow_types[workflow_type] = workflow_types.get(workflow_type, 0) + 1
        
        return {
            "total_sessions": len(sessions),
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "average_messages_per_session": total_messages / len(sessions) if sessions else 0,
            "workflow_distribution": workflow_types,
            "available_workflows": list(self.workflow_functions.keys())
        }
