"""
Model Manager for handling HuggingFace models and LangChain integration.

This module provides comprehensive AI model management capabilities including:
- HuggingFace API integration for cloud-based models
- Local model loading and pipeline creation
- LangChain conversation chain management
- Response generation with memory and context
- Error handling and performance monitoring

Key Classes:
- ModelManager: Main interface for model operations
- APIModelManager: Direct API calls without LangChain
- ModelResponse: Structured response format

The module supports multiple model providers and handles both API-based and local models.
"""

# Import warning suppression first to avoid TensorFlow/PyTorch warnings
import sys
import os
from pathlib import Path

# Warning suppression is handled in config.py, no need to import suppress_warnings here

import time
import requests
import streamlit as st
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
from datetime import datetime

# LangChain imports for conversation management
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Transformers imports for model loading
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig,
    pipeline, infer_device
)
import torch

# Enhanced tokenizer and image processor managers
from .tokenizer_manager import tokenizer_manager
from .image_processor_manager import image_processor_manager

# Import configuration and logging
from .config import Config, ModelConfig, ModelProvider

# Import logging with fallback
try:
    from .logger import debug, info, warning, error, log_model_operation, log_api_call, time_operation, LoggedOperation
except ImportError:
    # Fallback logging functions
    def debug(msg, component="models", **kwargs): pass
    def info(msg, component="models", **kwargs): pass
    def warning(msg, component="models", **kwargs): pass
    def error(msg, component="models", error_obj=None, **kwargs): pass
    def log_model_operation(op, model=None, **kwargs): pass
    def log_api_call(endpoint, method="POST", **kwargs): pass
    def time_operation(name, component): 
        def decorator(func): return func
        return decorator
    
    class LoggedOperation:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass

# Import vision model manager
try:
    from .vision_model_manager import VisionModelManager, create_vision_manager
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    debug("Vision model manager not available", "models")

# Import HuggingFace Hub for model sharing
try:
    from huggingface_hub import HfApi, create_repo, login as hf_login, whoami
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    debug("HuggingFace Hub not available", "models")


@dataclass
class ModelResponse:
    """
    Structured response format from AI models.
    
    This class standardizes the response format across different model types and providers.
    It includes success/failure status, content, error information, and metadata for debugging.
    
    Attributes:
        content (str): The generated text response from the model
        success (bool): Whether the generation was successful
        error (Optional[str]): Error message if generation failed
        metadata (Optional[Dict[str, Any]]): Additional information like model name, timing, etc.
    """
    content: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelManager:
    """
    Manages AI models and provides unified interface for model operations.
    
    This class serves as the central hub for all AI model operations in DurgasAI:
    - Initializes and manages HuggingFace models (both API and local)
    - Creates LangChain conversation chains with memory
    - Handles response generation with error recovery
    - Manages chat history and session persistence
    - Provides performance monitoring and logging
    
    Supported Model Types:
    - HuggingFace API models (cloud-based, fast)
    - HuggingFace Local models (on-device, private)
    - Future: OpenAI, Anthropic, and other providers
    
    Architecture:
    - Uses LangChain for conversation management and memory
    - Integrates with Streamlit for caching and UI feedback
    - Implements comprehensive error handling and logging
    """
    
    def __init__(self):
        """
        Initialize the ModelManager with comprehensive setup.
        
        This constructor sets up the complete model management system including:
        - Model instance tracking and lifecycle management
        - LangChain integration for conversation management
        - Memory and chat history management
        - Performance monitoring and analytics
        - Error handling and recovery mechanisms
        
        State Components Initialized:
        - current_model: Raw model instance (for local models)
        - current_tokenizer: Tokenizer for the current model
        - chat_model: LangChain-wrapped model for conversation
        - chat_chain: Complete conversation chain with memory
        - chat_history: In-memory conversation history
        - Performance tracking variables for optimization
        
        Integration Features:
        - LangChain conversation chains with memory
        - HuggingFace API and local model support
        - Vision model integration (if available)
        - Model sharing capabilities (if available)
        - Comprehensive error handling and logging
        """
        info("Starting ModelManager initialization", "models")
        debug("ModelManager initialization beginning", "models")
        
        # Initialize model state tracking variables
        debug("Setting up model state tracking", "models")
        self.current_model = None          # Raw model instance (for local models)
        self.current_tokenizer = None      # Tokenizer for the current model
        self.chat_model = None             # LangChain-wrapped model
        self.chat_chain = None             # Complete conversation chain with memory
        self.chat_history = InMemoryChatMessageHistory()  # Conversation history
        debug("Model state variables initialized", "models",
              current_model=self.current_model is not None,
              current_tokenizer=self.current_tokenizer is not None,
              chat_history_length=len(self.chat_history.messages))
        
        # Vision model support initialization
        debug("Setting up vision model support", "models", vision_available=VISION_AVAILABLE)
        self.vision_manager = None         # Vision model manager instance
        self.vision_enabled = VISION_AVAILABLE  # Whether vision models are available
        debug("Vision model support configured", "models", vision_enabled=self.vision_enabled)
        
        # Model sharing support initialization
        debug("Setting up model sharing capabilities", "models", hf_hub_available=HF_HUB_AVAILABLE)
        self.hf_api = None                 # HuggingFace API instance
        self.sharing_enabled = HF_HUB_AVAILABLE  # Whether model sharing is available
        self.current_user = None           # Current HuggingFace user info
        debug("Model sharing support configured", "models", sharing_enabled=self.sharing_enabled)
        
        # Performance and state tracking
        self.model_load_time = None        # Time taken to load the current model
        self.last_response_time = None     # Time taken for last response generation
        self.total_responses = 0           # Total responses generated in this session
        self.failed_responses = 0          # Failed response attempts
        
        info("ModelManager initialized successfully", "models", 
             initial_state={
                "model_loaded": False,
                "chat_history_size": len(self.chat_history.messages),
                "total_responses": self.total_responses
            })
        
        log_model_operation("manager_initialized")
    
    @st.cache_resource
    def load_local_model(_self, model_config: ModelConfig) -> tuple:
        """
        Load local HuggingFace model with modern Transformers features and comprehensive logging.
        
        This method implements the latest best practices from Transformers documentation:
        1. Uses device_map="auto" and dtype="auto" for optimal loading
        2. Supports different model architectures (causal LM, general models)
        3. Implements proper error handling and fallback strategies
        4. Handles tokenizer configuration with modern features
        
        Args:
            model_config (ModelConfig): Configuration object containing model details
            
        Returns:
            tuple: (model, tokenizer, pipeline) if successful, (None, None, None) if failed
            
        Note: This method is cached by Streamlit to avoid reloading models
        """
        start_time = time.time()
        info(f"Loading local model: {model_config.model_id}", "models")
        
        try:
            # Step 1: Detect optimal device using Transformers utility
            device = infer_device()
            debug(f"Detected optimal device: {device}", "models")
            
            # Step 2: Load tokenizer using enhanced tokenizer manager
            debug(f"Loading tokenizer for {model_config.model_id}", "models")
            hf_kwargs = Config.get_huggingface_download_kwargs()
            
            # Use enhanced tokenizer manager for better caching and performance
            tokenizer_info = tokenizer_manager.load_tokenizer(
                model_config.model_id,
                **hf_kwargs
            )
            
            if not tokenizer_info:
                raise Exception(f"Failed to load tokenizer for {model_config.model_id}")
            
            tokenizer = tokenizer_info.tokenizer
            info(f"Enhanced tokenizer loaded successfully: {tokenizer.__class__.__name__}", "models",
                 vocab_size=tokenizer_info.vocab_size,
                 is_fast=tokenizer_info.is_fast,
                 load_time=tokenizer_info.load_time)
            
            # Step 3: Try loading as causal LM first (most common for chat)
            model = None
            model_type = None
            
            try:
                debug(f"Attempting to load as causal LM model", "models")
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_id,
                    device_map="auto",  # Optimal device allocation
                    dtype="auto",       # Optimal data type
                    cache_dir=Config.HUGGINGFACE_CACHE_DIR,
                    local_files_only=Config.HUGGINGFACE_LOCAL_FILES_ONLY,
                    trust_remote_code=False,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
                )
                model_type = "causal_lm"
                info("Model loaded as causal LM successfully", "models")
                
            except Exception as causal_error:
                debug(f"Failed to load as causal LM, trying general model: {causal_error}", "models")
                try:
                    model = AutoModel.from_pretrained(
                        model_config.model_id,
                        device_map="auto",
                        dtype="auto",
                        cache_dir=Config.HUGGINGFACE_CACHE_DIR,
                        local_files_only=Config.HUGGINGFACE_LOCAL_FILES_ONLY,
                        trust_remote_code=False,
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    model_type = "general"
                    info("Model loaded as general model successfully", "models")
                except Exception as general_error:
                    raise Exception(f"Failed to load model as both causal LM and general model: {general_error}")
            
            # Step 4: Tokenizer configuration handled by enhanced tokenizer manager
            debug("Tokenizer special tokens configured by enhanced manager", "models",
                  special_tokens=tokenizer_info.special_tokens)
            
            # Step 5: Create pipeline for easy inference
            debug(f"Creating pipeline for {model_config.model_id}", "models")
            
            # Determine the best pipeline task based on model type
            if model_type == "causal_lm":
                task = "text-generation"
            else:
                # Try to determine task from model config
                try:
                    config = AutoConfig.from_pretrained(model_config.model_id)
                    if hasattr(config, 'architectures') and config.architectures:
                        arch = config.architectures[0].lower()
                        if 'bert' in arch or 'roberta' in arch:
                            task = "feature-extraction"
                        else:
                            task = "text-generation"
                    else:
                        task = "text-generation"
                except:
                    task = "text-generation"
            
            # Create pipeline with appropriate parameters
            pipeline_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "device_map": "auto"
            }
            
            if task == "text-generation":
                pipeline_kwargs.update({
                    "max_new_tokens": model_config.max_tokens,
                    "temperature": model_config.temperature,
                    "top_p": model_config.top_p,
                    "repetition_penalty": model_config.repetition_penalty,
                    "do_sample": True,
                    "return_full_text": False
                })
            
            pipe = pipeline(task, **pipeline_kwargs)
            info(f"Pipeline created successfully for task: {task}", "models")
            
            # Log successful model loading
            load_time = time.time() - start_time
            device_info = {
                "detected_device": device,
                "model_device": str(model.device),
                "model_dtype": str(model.dtype),
                "model_type": model_type,
                "pipeline_task": task
            }
            
            info(f"Local model loaded successfully in {load_time:.2f}s", "models", 
                 model_id=model_config.model_id,
                 load_time=load_time,
                 device_info=device_info,
                 model_size=f"{model.num_parameters():,}" if hasattr(model, 'num_parameters') else "unknown")
            
            log_model_operation("local_model_loaded", model_config.model_id,
                load_time=load_time,
                device_info=device_info)
            
            return model, tokenizer, pipe
            
        except Exception as e:
            load_time = time.time() - start_time
            error_msg = f"Error loading local model {model_config.model_id}: {str(e)}"
            error(error_msg, "models", e, 
                  model_id=model_config.model_id,
                  load_time=load_time)
            
            st.error(error_msg)
            log_model_operation("local_model_load_failed", model_config.model_id,
                error=str(e),
                load_time=load_time)
            
            return None, None, None
    
    def setup_api_model(self, model_config: ModelConfig, api_token: str) -> Optional[ChatHuggingFace]:
        """
        Setup HuggingFace API model with LangChain integration.
        
        This method creates a cloud-based model connection:
        1. Validates API token format
        2. Creates HuggingFace endpoint with model configuration
        3. Wraps endpoint with ChatHuggingFace for conversation support
        4. Logs the setup process for debugging
        
        Args:
            model_config (ModelConfig): Model configuration with parameters
            api_token (str): HuggingFace API token for authentication
            
        Returns:
            Optional[ChatHuggingFace]: Configured chat model or None if failed
        """
        start_time = time.time()
        info(f"Setting up API model: {model_config.model_id}", "models")
        
        try:
            # Step 1: Validate API token format
            if not api_token or not api_token.startswith('hf_'):
                warning("Invalid API token format", "models", token_prefix=api_token[:10] if api_token else "empty")
                raise ValueError("Invalid HuggingFace API token format")
            
            debug(f"API token validated for {model_config.model_id}", "models")
            
            # Step 2: Create HuggingFace endpoint with model parameters
            debug(f"Creating HuggingFace endpoint with parameters", "models", 
                  model_id=model_config.model_id,
                  max_tokens=model_config.max_tokens,
                  temperature=model_config.temperature,
                  top_p=model_config.top_p,
                  repetition_penalty=model_config.repetition_penalty)
            
            llm = HuggingFaceEndpoint(
                repo_id=model_config.model_id,
                task="text-generation",
                max_new_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                repetition_penalty=model_config.repetition_penalty,
                huggingfacehub_api_token=api_token,
                do_sample=True
            )
            
            info("HuggingFace endpoint created successfully", "models")
            
            # Step 3: Wrap with ChatHuggingFace for chat functionality
            debug("Wrapping endpoint with ChatHuggingFace", "models")
            chat_model = ChatHuggingFace(llm=llm)
            
            # Log successful setup
            setup_time = time.time() - start_time
            info(f"API model setup completed in {setup_time:.2f}s", "models", 
                 model_id=model_config.model_id,
                 setup_time=setup_time,
                 provider=model_config.provider.value)
            
            log_model_operation("api_model_setup", model_config.model_id,
                setup_time=setup_time,
                success=True)
            
            return chat_model
            
        except Exception as e:
            setup_time = time.time() - start_time
            error_msg = f"Error setting up API model {model_config.model_id}: {str(e)}"
            error(error_msg, "models", e, 
                  model_id=model_config.model_id,
                  setup_time=setup_time,
                  api_token_present=bool(api_token))
            
            st.error(error_msg)
            log_model_operation("api_model_setup_failed", model_config.model_id,
                error=str(e),
                setup_time=setup_time)
            
            return None
    
    def setup_local_pipeline_model(self, model_config: ModelConfig) -> Optional[ChatHuggingFace]:
        """Setup local model with pipeline and LangChain."""
        try:
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model_config.model_id,
                max_new_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                repetition_penalty=model_config.repetition_penalty,
                do_sample=True,
                return_full_text=False,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Create HuggingFace pipeline wrapper
            llm = HuggingFacePipeline(pipeline=pipe)
            
            # Wrap with ChatHuggingFace
            chat_model = ChatHuggingFace(llm=llm)
            return chat_model
            
        except Exception as e:
            st.error(f"Error setting up local pipeline model: {str(e)}")
            return None
    
    def create_chat_chain(self, chat_model, system_prompt: str):
        """Create chat chain with memory."""
        try:
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            
            # Create chain
            chain = prompt | chat_model
            
            # Add memory
            chain_with_history = RunnableWithMessageHistory(
                chain,
                lambda session_id: self.chat_history,
                input_messages_key="input",
                history_messages_key="history",
            )
            
            return chain_with_history
            
        except Exception as e:
            st.error(f"Error creating chat chain: {str(e)}")
            return None
    
    @time_operation("model_initialization", "models")
    def initialize_model(self, model_name: str, api_token: str = "", system_prompt: str = "") -> bool:
        """
        Initialize the selected AI model with comprehensive logging and validation.
        
        This is the main method for setting up AI models in DurgasAI:
        1. Validates model configuration exists
        2. Resets chat history for fresh conversation
        3. Initializes model based on provider type (API vs Local)
        4. Creates LangChain conversation chain with memory
        5. Validates successful initialization
        6. Logs all operations for debugging
        
        Args:
            model_name (str): Key name of the model to initialize
            api_token (str): HuggingFace API token (required for API models)
            system_prompt (str): Custom system prompt (optional)
            
        Returns:
            bool: True if model initialized successfully, False otherwise
        """
        start_time = time.time()
        
        info(f"Initializing model: {model_name}", "models", 
             model_name=model_name,
             has_api_token=bool(api_token),
             has_custom_prompt=bool(system_prompt),
             current_model=st.session_state.get("current_model"))
        
        # Step 1: Validate model configuration exists
        debug(f"Looking up model configuration for: {model_name}", "models")
        model_config = Config.get_model_config(model_name)
        
        if not model_config:
            error_msg = f"Model configuration not found: {model_name}"
            error(error_msg, "models", extra_data={
                "requested_model": model_name,
                "available_models": list(Config.AVAILABLE_MODELS.keys())
            })
            
            st.error(error_msg)
            log_model_operation("model_init_failed", model_name, error="config_not_found")
            return False
        
        debug(f"Model configuration found: {model_config.name}", "models", 
              model_id=model_config.model_id,
              provider=model_config.provider.value,
              max_tokens=model_config.max_tokens,
              temperature=model_config.temperature)
        
        try:
            with LoggedOperation(f"initialize_model_{model_name}", "models", extra_data={"model_id": model_config.model_id}):
                
                # Step 2: Reset chat history for fresh conversation
                debug("Resetting chat history for new model", "models")
                previous_history_size = len(self.chat_history.messages)
                self.chat_history = InMemoryChatMessageHistory()
                
                if previous_history_size > 0:
                    info(f"Chat history reset: cleared {previous_history_size} messages", "models")
                
                # Step 3: Initialize model based on provider type
                debug(f"Initializing {model_config.provider.value} model", "models")
                
                if model_config.provider == ModelProvider.HUGGINGFACE_API:
                    # API model initialization
                    if not api_token:
                        error_msg = "HuggingFace API token required for API models"
                        warning(error_msg, "models", model_id=model_config.model_id)
                        st.error(error_msg)
                        log_model_operation("model_init_failed", model_name, error="no_api_token")
                        return False
                    
                    info("Setting up API model", "models")
                    self.chat_model = self.setup_api_model(model_config, api_token)
                    
                elif model_config.provider == ModelProvider.HUGGINGFACE_LOCAL:
                    # Local model initialization
                    info("Setting up local model", "models")
                    self.chat_model = self.setup_local_pipeline_model(model_config)
                
                else:
                    error_msg = f"Unsupported model provider: {model_config.provider}"
                    error(error_msg, "models")
                    st.error(error_msg)
                    return False
                
                # Step 4: Validate model setup
                if not self.chat_model:
                    error_msg = f"Failed to setup {model_config.provider.value} model"
                    error(error_msg, "models", extra_data={"model_config": model_config.__dict__})
                    log_model_operation("model_init_failed", model_name, error="model_setup_failed")
                    return False
                
                info("Chat model setup completed successfully", "models")
                
                # Step 5: Create conversation chain with memory
                debug("Creating conversation chain with memory", "models")
                
                if not system_prompt:
                    system_prompt = Config.DEFAULT_SYSTEM_PROMPTS["helpful_assistant"]
                    debug("Using default system prompt", "models")
                else:
                    debug("Using custom system prompt", "models", prompt_length=len(system_prompt))
                
                self.chat_chain = self.create_chat_chain(self.chat_model, system_prompt)
                
                # Step 6: Validate complete initialization
                if self.chat_chain:
                    initialization_time = time.time() - start_time
                    self.model_load_time = initialization_time
                    
                    # Update session state
                    if hasattr(st, 'session_state'):
                        st.session_state.model_loaded = True
                        st.session_state.current_model = model_config.name
                    
                    # Log successful initialization
                    info(f"Model initialized successfully: {model_config.name} in {initialization_time:.2f}s", "models",
                        model_name=model_config.name,
                        model_id=model_config.model_id,
                        provider=model_config.provider.value,
                        initialization_time=initialization_time,
                        system_prompt_length=len(system_prompt))
                    
                    log_model_operation("model_initialized", model_name,
                        initialization_time=initialization_time,
                        provider=model_config.provider.value,
                        success=True)
                    
                    st.success(f"✅ {model_config.name} loaded successfully!")
                    return True
                
                else:
                    error_msg = "Failed to create conversation chain"
                    error(error_msg, "models")
                    log_model_operation("model_init_failed", model_name, error="chain_creation_failed")
                    return False
            
        except Exception as e:
            initialization_time = time.time() - start_time
            error_msg = f"Error initializing model {model_name}: {str(e)}"
            
            error(error_msg, "models", e,
                model_name=model_name,
                initialization_time=initialization_time,
                has_api_token=bool(api_token),
                model_config=model_config.__dict__ if model_config else None)
            
            st.error(error_msg)
            log_model_operation("model_init_failed", model_name,
                error=str(e),
                initialization_time=initialization_time)
            
            return False
    
    @time_operation("response_generation", "models")
    def generate_response(self, user_input: str, session_id: str = "default") -> ModelResponse:
        """
        Generate response from the current model with comprehensive logging and error handling.
        
        This method is the core of the AI conversation system:
        1. Validates that a model is loaded and ready
        2. Processes user input through the LangChain conversation chain
        3. Extracts and formats the AI response
        4. Handles errors gracefully with user-friendly messages
        5. Logs all operations for debugging and performance monitoring
        
        Args:
            user_input (str): The user's message/question
            session_id (str): Session identifier for conversation context
            
        Returns:
            ModelResponse: Structured response with content, success status, and metadata
        """
        start_time = time.time()
        
        debug(f"Generating response for user input", "models",
            input_length=len(user_input),
            session_id=session_id,
            chat_chain_available=bool(self.chat_chain),
            chat_history_size=len(self.chat_history.messages))
        
        # Step 1: Validate model availability
        if not self.chat_chain:
            warning("Response generation attempted with no model loaded", "models")
            log_model_operation("response_generation_failed", None,
                error="no_model_loaded",
                user_input_length=len(user_input))
            
            return ModelResponse(
                content="No model loaded. Please initialize a model first.",
                success=False,
                error="No model loaded"
            )
        
        try:
            # Step 2: Log input processing
            info(f"Processing user input through chat chain", "models",
                input_preview=user_input[:100] + "..." if len(user_input) > 100 else user_input,
                session_id=session_id,
                history_messages=len(self.chat_history.messages))
            
            # Step 3: Generate response using LangChain chat chain
            debug("Invoking chat chain for response generation", "models")
            response = self.chat_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            debug("Chat chain invocation completed", "models",
                response_type=type(response).__name__,
                has_content_attr=hasattr(response, 'content'))
            
            # Step 4: Extract content from response
            if hasattr(response, 'content'):
                content = response.content
                debug("Extracted content from response.content", "models")
            else:
                content = str(response)
                debug("Converted response to string", "models")
            
            # Step 5: Validate response content
            if not content or not content.strip():
                warning("Empty response generated from model", "models")
                content = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            # Step 6: Calculate performance metrics
            response_time = time.time() - start_time
            self.last_response_time = response_time
            self.total_responses += 1
            
            # Update session state with performance data
            if hasattr(st, 'session_state'):
                st.session_state.last_model_response_time = response_time
            
            # Step 7: Log successful response generation
            info(f"Response generated successfully in {response_time:.2f}s", "models",
                response_length=len(content),
                response_time=response_time,
                total_responses=self.total_responses,
                session_id=session_id,
                content_preview=content[:100] + "..." if len(content) > 100 else content)
            
            log_model_operation("response_generated", st.session_state.get("current_model"),
                response_time=response_time,
                input_length=len(user_input),
                output_length=len(content),
                success=True)
            
            return ModelResponse(
                content=content,
                success=True,
                metadata={
                    "session_id": session_id,
                    "response_time": response_time,
                    "model_name": st.session_state.get("current_model"),
                    "timestamp": datetime.now().isoformat(),
                    "total_responses": self.total_responses
                }
            )
            
        except Exception as e:
            # Step 8: Handle and log errors
            response_time = time.time() - start_time
            self.failed_responses += 1
            
            error_msg = f"Error generating response: {str(e)}"
            error(error_msg, "models", e,
                user_input_length=len(user_input),
                session_id=session_id,
                response_time=response_time,
                total_responses=self.total_responses,
                failed_responses=self.failed_responses,
                chat_history_size=len(self.chat_history.messages))
            
            log_model_operation("response_generation_failed", st.session_state.get("current_model"),
                error=str(e),
                response_time=response_time,
                input_length=len(user_input),
                success=False)
            
            return ModelResponse(
                content="I apologize, but I encountered an error while processing your request. Please try again.",
                success=False,
                error=error_msg,
                metadata={
                    "session_id": session_id,
                    "response_time": response_time,
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def clear_chat_history(self):
        """
        Clear the chat history and reset conversation context.
        
        This method resets the conversation memory while preserving the model state.
        It's useful for starting fresh conversations or clearing sensitive data.
        """
        previous_message_count = len(self.chat_history.messages)
        debug(f"Clearing chat history with {previous_message_count} messages", "models")
        
        # Reset the chat history
        self.chat_history = InMemoryChatMessageHistory()
        
        info(f"Chat history cleared: removed {previous_message_count} messages", "models")
        log_model_operation("chat_history_cleared", st.session_state.get("current_model"),
            previous_message_count=previous_message_count)
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history."""
        history = []
        for message in self.chat_history.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history
    
    def initialize_vision_manager(self, api_token: str = None) -> bool:
        """
        Initialize the vision model manager.
        
        Args:
            api_token (str): HuggingFace API token for vision models
            
        Returns:
            bool: True if vision manager initialized successfully, False otherwise
        """
        if not self.vision_enabled:
            warning("Vision models not available - missing dependencies", "models")
            return False
        
        try:
            debug("Initializing vision model manager", "models")
            self.vision_manager = create_vision_manager(api_token)
            
            if self.vision_manager:
                info("Vision model manager initialized successfully", "models")
                log_model_operation("vision_manager_initialized")
                return True
            else:
                warning("Failed to initialize vision model manager", "models")
                return False
                
        except Exception as e:
            error(f"Error initializing vision manager: {str(e)}", "models", e)
            return False
    
    def analyze_image(self, image_path: str, text_prompt: str, 
                     model_id: str = "HuggingFaceM4/idefics2-8b") -> ModelResponse:
        """
        Analyze image with text prompt using vision model.
        
        Args:
            image_path (str): Path to the image file or URL
            text_prompt (str): Text prompt describing what to analyze
            model_id (str): HuggingFace model ID to use
            
        Returns:
            ModelResponse: Structured response with analysis results
        """
        if not self.vision_enabled:
            return ModelResponse(
                content="Vision models not available. Please install required dependencies.",
                success=False,
                error="Vision models not available"
            )
        
        if not self.vision_manager:
            # Try to initialize vision manager
            if not self.initialize_vision_manager():
                return ModelResponse(
                    content="Failed to initialize vision model manager. Please check your API token.",
                    success=False,
                    error="Vision manager initialization failed"
                )
        
        try:
            vision_response = self.vision_manager.analyze_image(image_path, text_prompt, model_id)
            
            return ModelResponse(
                content=vision_response.content,
                success=vision_response.success,
                error=vision_response.error,
                metadata=vision_response.metadata
            )
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            error(error_msg, "models", e)
            
            return ModelResponse(
                content="",
                success=False,
                error=error_msg
            )
    
    def generate_image_caption(self, image_path: str, 
                              model_id: str = "Salesforce/blip-image-captioning-base") -> ModelResponse:
        """
        Generate caption for image using specialized captioning model.
        
        Args:
            image_path (str): Path to the image file or URL
            model_id (str): HuggingFace model ID for captioning
            
        Returns:
            ModelResponse: Structured response with caption
        """
        if not self.vision_enabled:
            return ModelResponse(
                content="Vision models not available. Please install required dependencies.",
                success=False,
                error="Vision models not available"
            )
        
        if not self.vision_manager:
            if not self.initialize_vision_manager():
                return ModelResponse(
                    content="Failed to initialize vision model manager. Please check your API token.",
                    success=False,
                    error="Vision manager initialization failed"
                )
        
        try:
            vision_response = self.vision_manager.generate_caption(image_path, model_id)
            
            return ModelResponse(
                content=vision_response.content,
                success=vision_response.success,
                error=vision_response.error,
                metadata=vision_response.metadata
            )
            
        except Exception as e:
            error_msg = f"Error generating caption: {str(e)}"
            error(error_msg, "models", e)
            
            return ModelResponse(
                content="",
                success=False,
                error=error_msg
            )
    
    def visual_question_answering(self, image_path: str, question: str,
                                 model_id: str = "Salesforce/blip-vqa-base") -> ModelResponse:
        """
        Answer questions about images using VQA model.
        
        Args:
            image_path (str): Path to the image file or URL
            question (str): Question to ask about the image
            model_id (str): HuggingFace model ID for VQA
            
        Returns:
            ModelResponse: Structured response with answer
        """
        if not self.vision_enabled:
            return ModelResponse(
                content="Vision models not available. Please install required dependencies.",
                success=False,
                error="Vision models not available"
            )
        
        if not self.vision_manager:
            if not self.initialize_vision_manager():
                return ModelResponse(
                    content="Failed to initialize vision model manager. Please check your API token.",
                    success=False,
                    error="Vision manager initialization failed"
                )
        
        try:
            vision_response = self.vision_manager.visual_question_answering(image_path, question, model_id)
            
            return ModelResponse(
                content=vision_response.content,
                success=vision_response.success,
                error=vision_response.error,
                metadata=vision_response.metadata
            )
            
        except Exception as e:
            error_msg = f"Error answering visual question: {str(e)}"
            error(error_msg, "models", e)
            
            return ModelResponse(
                content="",
                success=False,
                error=error_msg
            )
    
    def get_vision_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for vision models.
        
        Returns:
            Dict[str, Any]: Vision model performance metrics
        """
        if not self.vision_manager:
            return {"error": "Vision manager not initialized"}
        
        return self.vision_manager.get_performance_metrics()
    
    # ===== MODEL SHARING METHODS =====
    
    def initialize_sharing_manager(self, api_token: str = None) -> bool:
        """
        Initialize the HuggingFace Hub sharing manager.
        
        Args:
            api_token (str): HuggingFace API token for sharing models
            
        Returns:
            bool: True if sharing manager initialized successfully, False otherwise
        """
        if not self.sharing_enabled:
            warning("Model sharing not available - missing dependencies", "models")
            return False
        
        try:
            debug("Initializing HuggingFace sharing manager", "models")
            
            # Initialize HfApi
            if api_token:
                self.hf_api = HfApi(token=api_token)
                debug("HuggingFace API initialized with token", "models")
            else:
                self.hf_api = HfApi()
                debug("HuggingFace API initialized without token", "models")
            
            # Get current user info
            try:
                self.current_user = whoami(token=api_token)
                info(f"HuggingFace sharing initialized for user: {self.current_user['name']}", "models")
                log_model_operation("sharing_manager_initialized", user=self.current_user['name'])
                return True
            except Exception as e:
                warning(f"Could not get user info: {e}", "models")
                info("HuggingFace sharing initialized without user authentication", "models")
                return True
                
        except Exception as e:
            error(f"Error initializing sharing manager: {str(e)}", "models", e)
            return False
    
    def share_model(self, model_path: str, repo_id: str, 
                   private: bool = False, gated: bool = False,
                   license: str = "apache-2.0", 
                   commit_message: str = "Add model files") -> ModelResponse:
        """
        Share a model to HuggingFace Hub.
        
        Args:
            model_path (str): Path to the model directory
            repo_id (str): Repository ID for the model (e.g., "username/model-name")
            private (bool): Whether the repository should be private
            gated (bool): Whether the model should be gated
            license (str): License for the model
            commit_message (str): Commit message for the upload
            
        Returns:
            ModelResponse: Structured response with sharing results
        """
        if not self.sharing_enabled:
            return ModelResponse(
                content="Model sharing not available. Please install huggingface_hub.",
                success=False,
                error="Sharing not available"
            )
        
        if not self.hf_api:
            if not self.initialize_sharing_manager():
                return ModelResponse(
                    content="Failed to initialize sharing manager.",
                    success=False,
                    error="Sharing manager initialization failed"
                )
        
        try:
            debug(f"Starting model sharing process", "models", 
                  repo_id=repo_id, model_path=model_path, private=private)
            
            # Step 1: Create repository
            info(f"Creating repository: {repo_id}", "models")
            try:
                create_repo(
                    repo_id=repo_id,
                    exist_ok=True,
                    private=private,
                    token=self.hf_api.token
                )
                info(f"Repository created successfully: {repo_id}", "models")
            except Exception as e:
                if "already exists" in str(e).lower():
                    info(f"Repository already exists: {repo_id}", "models")
                else:
                    raise e
            
            # Step 2: Upload model files
            info(f"Uploading model files from: {model_path}", "models")
            result = self.hf_api.upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                create_commits=True
            )
            
            # Step 3: Update repository settings if needed
            if gated:
                info(f"Setting repository as gated: {repo_id}", "models")
                # Note: Gating requires additional API calls that may need special permissions
            
            # Log successful sharing
            hub_url = f"https://huggingface.co/{repo_id}"
            info(f"Model shared successfully: {hub_url}", "models")
            
            log_model_operation("model_shared", repo_id=repo_id,
                private=private, gated=gated, license=license)
            
            return ModelResponse(
                content=f"Model shared successfully! View at: {hub_url}",
                success=True,
                metadata={
                    "repo_id": repo_id,
                    "hub_url": hub_url,
                    "private": private,
                    "gated": gated,
                    "license": license
                }
            )
            
        except Exception as e:
            error_msg = f"Error sharing model: {str(e)}"
            error(error_msg, "models", e, repo_id=repo_id)
            
            return ModelResponse(
                content="Failed to share model. Please check your authentication and try again.",
                success=False,
                error=error_msg,
                metadata={"repo_id": repo_id}
            )
    
    def create_model_card(self, model_name: str, description: str,
                         performance_metrics: Dict[str, Any] = None,
                         usage_example: str = None,
                         limitations: str = None,
                         license: str = "apache-2.0") -> str:
        """
        Create a model card (README.md) for the model.
        
        Args:
            model_name (str): Name of the model
            description (str): Description of the model
            performance_metrics (Dict[str, Any]): Performance metrics
            usage_example (str): Usage example code
            limitations (str): Model limitations and bias information
            license (str): License for the model
            
        Returns:
            str: Generated model card content
        """
        try:
            debug(f"Creating model card for: {model_name}", "models")
            
            # Build model card content
            model_card = f"""---
license: {license}
language:
- en
pipeline_tag: text-generation
tags:
- text-generation
- transformers
- pytorch
---

# {model_name}

## Model Description

{description}

## Model Performance

"""
            
            # Add performance metrics if provided
            if performance_metrics:
                model_card += "| Metric | Value |\n|--------|-------|\n"
                for metric, value in performance_metrics.items():
                    model_card += f"| {metric} | {value} |\n"
            else:
                model_card += "Performance metrics not available.\n"
            
            model_card += "\n## Usage\n\n"
            
            # Add usage example
            if usage_example:
                model_card += f"```python\n{usage_example}\n```\n\n"
            else:
                model_card += """```python
from transformers import pipeline

generator = pipeline('text-generation', model='username/model-name')
result = generator("Hello, I am")
```\n\n"""
            
            # Add limitations if provided
            if limitations:
                model_card += f"## Limitations and Bias\n\n{limitations}\n\n"
            else:
                model_card += """## Limitations and Bias

- The model may exhibit bias present in the training data
- Performance may vary across different domains
- Limited to English language processing\n\n"""
            
            model_card += """## Ethical Considerations

Users should be aware of potential biases and limitations when using this model.

## Citation

```bibtex
@misc{""" + model_name.lower().replace(" ", "-") + """,
  title={""" + model_name + """},
  author={Your Name},
  year={2024},
  publisher={Hugging Face}
}
```"""
            
            info(f"Model card created for: {model_name}", "models")
            return model_card
            
        except Exception as e:
            error_msg = f"Error creating model card: {str(e)}"
            error(error_msg, "models", e, model_name=model_name)
            return f"# {model_name}\n\n{description}\n\nError creating detailed model card: {error_msg}"
    
    def get_sharing_status(self) -> Dict[str, Any]:
        """
        Get the current status of the sharing manager.
        
        Returns:
            Dict[str, Any]: Sharing manager status information
        """
        status = {
            "sharing_enabled": self.sharing_enabled,
            "hf_api_available": self.hf_api is not None,
            "user_authenticated": self.current_user is not None,
            "current_user": self.current_user.get('name') if self.current_user else None
        }
        
        return status


class APIModelManager:
    """Direct API model manager for simpler API calls."""
    
    @staticmethod
    def query_huggingface_api(model_id: str, payload: Dict[str, Any], api_token: str) -> Dict[str, Any]:
        """Query HuggingFace Inference API directly."""
        headers = {"Authorization": f"Bearer {api_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        
        try:
            response = requests.post(
                api_url, 
                headers=headers, 
                json=payload, 
                timeout=Config.API_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
                
        except requests.exceptions.Timeout:
            return {"error": "Request timeout. Please try again."}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    @staticmethod
    def generate_with_api(model_config: ModelConfig, user_input: str, api_token: str, 
                         conversation_history: str = "") -> ModelResponse:
        """Generate response using direct API call."""
        try:
            # Prepare input
            full_prompt = f"{conversation_history}Human: {user_input}\nAssistant:"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": model_config.max_tokens,
                    "temperature": model_config.temperature,
                    "top_p": model_config.top_p,
                    "repetition_penalty": model_config.repetition_penalty,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            result = APIModelManager.query_huggingface_api(
                model_config.model_id, payload, api_token
            )
            
            if "error" in result:
                return ModelResponse(
                    content="I apologize, but I encountered an error. Please try again.",
                    success=False,
                    error=result["error"]
                )
            
            # Extract response
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "").strip()
            elif isinstance(result, dict) and "generated_text" in result:
                content = result["generated_text"].strip()
            else:
                content = "I couldn't generate a proper response. Please try again."
            
            return ModelResponse(
                content=content,
                success=True,
                metadata={"model": model_config.model_id}
            )
            
        except Exception as e:
            return ModelResponse(
                content="I apologize, but I encountered an error. Please try again.",
                success=False,
                error=str(e)
            )
