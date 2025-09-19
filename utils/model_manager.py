"""
Model Manager for handling HuggingFace models and LangChain integration.
"""

# Import warning suppression first
import sys
import os
from pathlib import Path

# Add parent directory to path to import suppress_warnings
sys.path.insert(0, str(Path(__file__).parent.parent))
import suppress_warnings

import time
import requests
import streamlit as st
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

# LangChain imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Transformers imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from .config import Config, ModelConfig, ModelProvider


@dataclass
class ModelResponse:
    """Response from AI model."""
    content: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelManager:
    """Manages AI models and provides unified interface."""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.chat_model = None
        self.chat_chain = None
        self.chat_history = InMemoryChatMessageHistory()
    
    @st.cache_resource
    def load_local_model(_self, model_config: ModelConfig) -> tuple:
        """Load local HuggingFace model."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_config.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading local model: {str(e)}")
            return None, None
    
    def setup_api_model(self, model_config: ModelConfig, api_token: str) -> Optional[ChatHuggingFace]:
        """Setup HuggingFace API model with LangChain."""
        try:
            # Create HuggingFace endpoint
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
            
            # Wrap with ChatHuggingFace for chat functionality
            chat_model = ChatHuggingFace(llm=llm)
            return chat_model
            
        except Exception as e:
            st.error(f"Error setting up API model: {str(e)}")
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
    
    def initialize_model(self, model_name: str, api_token: str = "", system_prompt: str = "") -> bool:
        """Initialize the selected model."""
        model_config = Config.get_model_config(model_name)
        if not model_config:
            st.error(f"Model configuration not found: {model_name}")
            return False
        
        try:
            # Reset chat history
            self.chat_history = InMemoryChatMessageHistory()
            
            if model_config.provider == ModelProvider.HUGGINGFACE_API:
                if not api_token:
                    st.error("HuggingFace API token required for API models")
                    return False
                
                self.chat_model = self.setup_api_model(model_config, api_token)
                
            elif model_config.provider == ModelProvider.HUGGINGFACE_LOCAL:
                self.chat_model = self.setup_local_pipeline_model(model_config)
            
            if self.chat_model:
                # Create chat chain
                if not system_prompt:
                    system_prompt = Config.DEFAULT_SYSTEM_PROMPTS["helpful_assistant"]
                
                self.chat_chain = self.create_chat_chain(self.chat_model, system_prompt)
                
                if self.chat_chain:
                    st.success(f"✅ {model_config.name} loaded successfully!")
                    return True
            
            return False
            
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return False
    
    def generate_response(self, user_input: str, session_id: str = "default") -> ModelResponse:
        """Generate response from the current model."""
        if not self.chat_chain:
            return ModelResponse(
                content="No model loaded. Please initialize a model first.",
                success=False,
                error="No model loaded"
            )
        
        try:
            # Generate response using chat chain
            response = self.chat_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Extract content from response
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            return ModelResponse(
                content=content,
                success=True,
                metadata={"session_id": session_id}
            )
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            return ModelResponse(
                content="I apologize, but I encountered an error while processing your request. Please try again.",
                success=False,
                error=error_msg
            )
    
    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history = InMemoryChatMessageHistory()
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get formatted chat history."""
        history = []
        for message in self.chat_history.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history


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
