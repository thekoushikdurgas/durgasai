"""
Model Catalog Page for DurgasAI.

This module provides a comprehensive catalog of all Hugging Face AI models
organized by categories with detailed information about system requirements,
configurations, and download capabilities.
"""

import streamlit as st
import requests
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
from datetime import datetime

# Import utilities
try:
    from utils.logger import debug, info, warning, error, log_user_action
except ImportError:
    def debug(msg, component="model_catalog", **kwargs): pass
    def info(msg, component="model_catalog", **kwargs): pass
    def warning(msg, component="model_catalog", **kwargs): pass
    def error(msg, component="model_catalog", **kwargs): pass
    def log_user_action(action, **kwargs): pass


class ModelCatalogManager:
    """Manages the comprehensive model catalog with system requirements and configurations."""
    
    def __init__(self):
        """Initialize the model catalog manager."""
        self.catalog_data = self._load_catalog_data()
        self.api_base_url = "https://huggingface.co/api/models"
        self.cache_duration = 3600  # 1 hour cache
        self.last_cache_update = None
        
    def _load_catalog_data(self) -> Dict[str, Any]:
        """Load comprehensive model catalog data."""
        return {
            "multimodal": {
                "title": "Multimodal",
                "description": "Models that process multiple types of data inputs (text, images, audio, etc.)",
                "icon": "🎭",
                "model_count": 7698,
                "categories": {
                    "any_to_any": {
                        "name": "Any-to-Any",
                        "count": 7698,
                        "models": [
                            {
                                "id": "HuggingFaceM4/idefics2-8b",
                                "name": "Idefics2 8B",
                                "description": "Open multimodal model that accepts arbitrary sequences of image and text inputs",
                                "size": "8B",
                                "system_requirements": {
                                    "gpu_memory": "16GB+",
                                    "ram": "32GB+",
                                    "storage": "20GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["image_analysis", "visual_qa", "ocr", "document_understanding"],
                                "license": "apache-2.0"
                            },
                            {
                                "id": "HuggingFaceM4/idefics2-8b-chatty",
                                "name": "Idefics2 8B Chatty",
                                "description": "Idefics2-8b fine-tuned for extended conversations",
                                "size": "8B",
                                "system_requirements": {
                                    "gpu_memory": "16GB+",
                                    "ram": "32GB+",
                                    "storage": "20GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["detailed_image_analysis", "long_form_qa", "storytelling"],
                                "license": "apache-2.0"
                            }
                        ]
                    },
                    "audio_text_to_text": {
                        "name": "Audio-Text-to-Text",
                        "count": 126,
                        "models": [
                            {
                                "id": "openai/whisper-base",
                                "name": "Whisper Base",
                                "description": "Automatic speech recognition model",
                                "size": "74M",
                                "system_requirements": {
                                    "gpu_memory": "2GB+",
                                    "ram": "8GB+",
                                    "storage": "1GB",
                                    "cuda": False,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["speech_recognition", "transcription"],
                                "license": "mit"
                            }
                        ]
                    },
                    "document_qa": {
                        "name": "Document Question Answering",
                        "count": 233,
                        "models": []
                    },
                    "visual_document_retrieval": {
                        "name": "Visual Document Retrieval",
                        "count": 89,
                        "models": []
                    },
                    "image_text_to_text": {
                        "name": "Image-Text-to-Text",
                        "count": 6549,
                        "models": []
                    },
                    "video_text_to_text": {
                        "name": "Video-Text-to-Text",
                        "count": 177,
                        "models": []
                    },
                    "visual_qa": {
                        "name": "Visual Question Answering",
                        "count": 515,
                        "models": [
                            {
                                "id": "Salesforce/blip-vqa-base",
                                "name": "BLIP VQA Base",
                                "description": "Visual question answering model",
                                "size": "224M",
                                "system_requirements": {
                                    "gpu_memory": "4GB+",
                                    "ram": "8GB+",
                                    "storage": "1GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["visual_qa", "image_understanding"],
                                "license": "bsd-3-clause"
                            }
                        ]
                    }
                }
            },
            "natural_language_processing": {
                "title": "Natural Language Processing",
                "description": "Models for text-based tasks including classification, generation, and understanding",
                "icon": "📝",
                "model_count": 500000,  # Approximate total
                "categories": {
                    "feature_extraction": {
                        "name": "Feature Extraction",
                        "count": 14226,
                        "models": [
                            {
                                "id": "sentence-transformers/all-MiniLM-L6-v2",
                                "name": "All MiniLM L6 v2",
                                "description": "Universal sentence encoder",
                                "size": "22M",
                                "system_requirements": {
                                    "gpu_memory": "1GB+",
                                    "ram": "4GB+",
                                    "storage": "500MB",
                                    "cuda": False,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["embeddings", "similarity", "clustering"],
                                "license": "apache-2.0"
                            }
                        ]
                    },
                    "text_generation": {
                        "name": "Text Generation",
                        "count": 282414,
                        "models": [
                            {
                                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                                "name": "Mistral 7B Instruct",
                                "description": "Powerful instruction-following model",
                                "size": "7B",
                                "system_requirements": {
                                    "gpu_memory": "16GB+",
                                    "ram": "32GB+",
                                    "storage": "15GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["instruction_following", "reasoning", "coding"],
                                "license": "apache-2.0"
                            },
                            {
                                "id": "HuggingFaceH4/zephyr-7b-beta",
                                "name": "Zephyr 7B Beta",
                                "description": "Great conversational AI model",
                                "size": "7B",
                                "system_requirements": {
                                    "gpu_memory": "16GB+",
                                    "ram": "32GB+",
                                    "storage": "15GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["chat", "qa", "general_conversation"],
                                "license": "mit"
                            }
                        ]
                    },
                    "text_classification": {
                        "name": "Text Classification",
                        "count": 101608,
                        "models": [
                            {
                                "id": "distilbert-base-uncased-finetuned-sst-2-english",
                                "name": "DistilBERT SST-2",
                                "description": "Sentiment analysis model",
                                "size": "66M",
                                "system_requirements": {
                                    "gpu_memory": "1GB+",
                                    "ram": "4GB+",
                                    "storage": "500MB",
                                    "cuda": False,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["sentiment_analysis", "classification"],
                                "license": "apache-2.0"
                            }
                        ]
                    },
                    "question_answering": {
                        "name": "Question Answering",
                        "count": 13009,
                        "models": []
                    },
                    "summarization": {
                        "name": "Summarization",
                        "count": 2482,
                        "models": []
                    },
                    "translation": {
                        "name": "Translation",
                        "count": 7641,
                        "models": []
                    }
                }
            },
            "computer_vision": {
                "title": "Computer Vision",
                "description": "Models for image and video processing tasks",
                "icon": "👁️",
                "model_count": 150000,  # Approximate total
                "categories": {
                    "image_classification": {
                        "name": "Image Classification",
                        "count": 19564,
                        "models": [
                            {
                                "id": "google/vit-base-patch16-224",
                                "name": "ViT Base",
                                "description": "Vision Transformer for image classification",
                                "size": "86M",
                                "system_requirements": {
                                    "gpu_memory": "4GB+",
                                    "ram": "8GB+",
                                    "storage": "1GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["image_classification", "object_recognition"],
                                "license": "apache-2.0"
                            }
                        ]
                    },
                    "object_detection": {
                        "name": "Object Detection",
                        "count": 3750,
                        "models": [
                            {
                                "id": "facebook/detr-resnet-50",
                                "name": "DETR ResNet-50",
                                "description": "End-to-end object detection",
                                "size": "41M",
                                "system_requirements": {
                                    "gpu_memory": "6GB+",
                                    "ram": "8GB+",
                                    "storage": "1GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["object_detection", "instance_segmentation"],
                                "license": "apache-2.0"
                            }
                        ]
                    },
                    "text_to_image": {
                        "name": "Text-to-Image",
                        "count": 86483,
                        "models": [
                            {
                                "id": "runwayml/stable-diffusion-v1-5",
                                "name": "Stable Diffusion v1.5",
                                "description": "Text-to-image generation model",
                                "size": "860M",
                                "system_requirements": {
                                    "gpu_memory": "8GB+",
                                    "ram": "16GB+",
                                    "storage": "4GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["image_generation", "art_creation", "content_creation"],
                                "license": "creativeml-openrail-m"
                            }
                        ]
                    },
                    "image_to_text": {
                        "name": "Image-to-Text",
                        "count": 8644,
                        "models": [
                            {
                                "id": "Salesforce/blip-image-captioning-base",
                                "name": "BLIP Image Captioning",
                                "description": "Image captioning model",
                                "size": "224M",
                                "system_requirements": {
                                    "gpu_memory": "4GB+",
                                    "ram": "8GB+",
                                    "storage": "1GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["image_captioning", "description"],
                                "license": "bsd-3-clause"
                            }
                        ]
                    }
                }
            },
            "audio": {
                "title": "Audio",
                "description": "Models for audio processing, speech recognition, and generation",
                "icon": "🎵",
                "model_count": 50000,  # Approximate total
                "categories": {
                    "automatic_speech_recognition": {
                        "name": "Automatic Speech Recognition",
                        "count": 26043,
                        "models": [
                            {
                                "id": "openai/whisper-large-v3",
                                "name": "Whisper Large v3",
                                "description": "Large-scale automatic speech recognition",
                                "size": "1550M",
                                "system_requirements": {
                                    "gpu_memory": "8GB+",
                                    "ram": "16GB+",
                                    "storage": "3GB",
                                    "cuda": True,
                                    "os": ["Linux", "Windows", "macOS"]
                                },
                                "use_cases": ["speech_recognition", "transcription", "multilingual_asr"],
                                "license": "mit"
                            }
                        ]
                    },
                    "text_to_speech": {
                        "name": "Text-to-Speech",
                        "count": 3376,
                        "models": []
                    },
                    "audio_classification": {
                        "name": "Audio Classification",
                        "count": 3518,
                        "models": []
                    },
                    "audio_to_audio": {
                        "name": "Audio-to-Audio",
                        "count": 3966,
                        "models": []
                    }
                }
            },
            "reinforcement_learning": {
                "title": "Reinforcement Learning",
                "description": "Models that learn optimal behaviors through interactions",
                "icon": "🎮",
                "model_count": 64703,
                "categories": {
                    "reinforcement_learning": {
                        "name": "Reinforcement Learning",
                        "count": 64703,
                        "models": []
                    }
                }
            }
        }
    
    def get_model_info_from_api(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch model information from Hugging Face API."""
        try:
            response = requests.get(f"{self.api_base_url}/{model_id}", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                warning(f"Failed to fetch model info for {model_id}: {response.status_code}")
                return None
        except Exception as e:
            error(f"Error fetching model info for {model_id}: {str(e)}")
            return None
    
    def get_system_requirements(self, model_id: str, model_size: str) -> Dict[str, Any]:
        """Estimate system requirements based on model characteristics."""
        # Convert model size to parameters
        size_mapping = {
            "22M": 22000000,
            "66M": 66000000,
            "74M": 74000000,
            "86M": 86000000,
            "224M": 224000000,
            "860M": 860000000,
            "7B": 7000000000,
            "8B": 8000000000,
            "17B": 17000000000,
            "1550M": 1550000000
        }
        
        params = size_mapping.get(model_size, 1000000000)  # Default to 1B
        
        if params < 100_000_000:  # < 100M parameters
            return {
                "gpu_memory": "2GB+",
                "ram": "8GB+",
                "storage": "1GB",
                "cuda": False,
                "os": ["Linux", "Windows", "macOS"]
            }
        elif params < 1_000_000_000:  # < 1B parameters
            return {
                "gpu_memory": "4GB+",
                "ram": "16GB+",
                "storage": "2GB",
                "cuda": True,
                "os": ["Linux", "Windows", "macOS"]
            }
        elif params < 10_000_000_000:  # < 10B parameters
            return {
                "gpu_memory": "16GB+",
                "ram": "32GB+",
                "storage": "15GB",
                "cuda": True,
                "os": ["Linux", "Windows", "macOS"]
            }
        else:  # > 10B parameters
            return {
                "gpu_memory": "24GB+",
                "ram": "64GB+",
                "storage": "30GB",
                "cuda": True,
                "os": ["Linux", "Windows"]
            }


def render_model_catalog_page():
    """Render the comprehensive model catalog page."""
    st.markdown('<h1 class="main-header">🤖 Hugging Face Model Catalog</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Comprehensive AI Model Repository
    
    Explore the complete collection of Hugging Face AI models organized by categories.
    Each model includes detailed system requirements, configuration options, and download capabilities.
    """)
    
    # Initialize catalog manager
    catalog_manager = ModelCatalogManager()
    
    # Create sidebar filters
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        
        # Category filter
        categories = list(catalog_manager.catalog_data.keys())
        selected_category = st.selectbox(
            "Select Category:",
            ["All Categories"] + [cat.replace("_", " ").title() for cat in categories],
            index=0
        )
        
        # System requirements filter
        st.markdown("#### System Requirements")
        min_gpu_memory = st.selectbox(
            "Minimum GPU Memory:",
            ["Any", "2GB+", "4GB+", "8GB+", "16GB+", "24GB+"],
            index=0
        )
        
        cuda_required = st.checkbox("CUDA Required", value=False)
        
        # Model size filter
        model_sizes = st.multiselect(
            "Model Sizes:",
            ["22M", "66M", "74M", "86M", "224M", "860M", "7B", "8B", "17B", "1550M"],
            default=[]
        )
    
    # Main content area
    if selected_category == "All Categories":
        # Display all categories
        for category_key, category_data in catalog_manager.catalog_data.items():
            render_category_section(category_key, category_data, catalog_manager)
    else:
        # Display selected category
        category_key = selected_category.lower().replace(" ", "_")
        if category_key in catalog_manager.catalog_data:
            render_category_section(category_key, catalog_manager.catalog_data[category_key], catalog_manager)
    
    # Download section
    render_download_section()


def render_category_section(category_key: str, category_data: Dict[str, Any], catalog_manager: ModelCatalogManager):
    """Render a category section with all its subcategories and models."""
    st.markdown(f"""
    ### {category_data['icon']} {category_data['title']}
    
    **{category_data['description']}**
    
    Total Models: **{category_data['model_count']:,}**
    """)
    
    # Create tabs for subcategories
    subcategories = list(category_data['categories'].keys())
    if subcategories:
        tabs = st.tabs([subcat['name'] for subcat in category_data['categories'].values()])
        
        for i, (subcat_key, subcat_data) in enumerate(category_data['categories'].items()):
            with tabs[i]:
                render_subcategory_section(subcat_data, catalog_manager)
    
    st.markdown("---")


def render_subcategory_section(subcat_data: Dict[str, Any], catalog_manager: ModelCatalogManager):
    """Render a subcategory section with model cards."""
    st.markdown(f"**{subcat_data['name']}** ({subcat_data['count']:,} models)")
    
    if subcat_data['models']:
        # Create columns for model cards
        cols = st.columns(2)
        
        for i, model in enumerate(subcat_data['models']):
            with cols[i % 2]:
                render_model_card(model, catalog_manager)
    else:
        st.info(f"No detailed models available for {subcat_data['name']}. "
                f"Check the [Hugging Face Model Hub](https://huggingface.co/models) for more models.")


def render_model_card(model: Dict[str, Any], catalog_manager: ModelCatalogManager):
    """Render an individual model card with detailed information."""
    with st.container():
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h4 style="margin: 0; color: #2E8B57;">{model['name']}</h4>
            <p style="margin: 5px 0; font-size: 14px; color: #666;">{model['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model ID:** `{model['id']}`")
            st.markdown(f"**Size:** {model['size']}")
            st.markdown(f"**License:** {model['license']}")
        
        with col2:
            # System requirements
            reqs = model.get('system_requirements', {})
            st.markdown("**System Requirements:**")
            st.markdown(f"• GPU Memory: {reqs.get('gpu_memory', 'N/A')}")
            st.markdown(f"• RAM: {reqs.get('ram', 'N/A')}")
            st.markdown(f"• Storage: {reqs.get('storage', 'N/A')}")
            st.markdown(f"• CUDA: {'Yes' if reqs.get('cuda', False) else 'No'}")
        
        # Use cases
        if model.get('use_cases'):
            st.markdown("**Use Cases:**")
            use_cases = ", ".join(model['use_cases'])
            st.markdown(f"*{use_cases}*")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download", key=f"download_{model['id']}"):
                download_model(model, catalog_manager)
        
        with col2:
            if st.button("📋 Copy ID", key=f"copy_{model['id']}"):
                st.code(model['id'])
                st.success("Model ID copied to clipboard!")
        
        with col3:
            if st.button("🔗 View on HF", key=f"view_{model['id']}"):
                st.markdown(f"[Open on Hugging Face](https://huggingface.co/{model['id']})")
        
        st.markdown("---")


def download_model(model: Dict[str, Any], catalog_manager: ModelCatalogManager):
    """Handle model download functionality."""
    log_user_action("model_download_attempted", model_id=model['id'])
    
    with st.spinner(f"Downloading {model['name']}..."):
        try:
            # Create download directory
            download_dir = Path("output/downloaded_models")
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model info
            model_info_file = download_dir / f"{model['id'].replace('/', '_')}_info.json"
            with open(model_info_file, 'w', encoding='utf-8') as f:
                json.dump(model, f, indent=2, ensure_ascii=False)
            
            # Generate download script
            download_script = generate_download_script(model)
            script_file = download_dir / f"{model['id'].replace('/', '_')}_download.py"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(download_script)
            
            st.success(f"✅ Download prepared for {model['name']}")
            st.markdown(f"**Download Location:** `{download_dir}`")
            st.markdown(f"**Script Generated:** `{script_file.name}`")
            
            # Provide download instructions
            with st.expander("📋 Download Instructions"):
                st.markdown(f"""
                ### How to Download and Use {model['name']}
                
                1. **Install Required Dependencies:**
                   ```bash
                   pip install transformers torch torchvision
                   ```
                
                2. **Run the Download Script:**
                   ```bash
                   python {script_file.name}
                   ```
                
                3. **System Requirements:**
                   - GPU Memory: {model.get('system_requirements', {}).get('gpu_memory', 'N/A')}
                   - RAM: {model.get('system_requirements', {}).get('ram', 'N/A')}
                   - Storage: {model.get('system_requirements', {}).get('storage', 'N/A')}
                
                4. **Usage Example:**
                   ```python
                   from transformers import AutoModel, AutoTokenizer
                   
                   model_name = "{model['id']}"
                   model = AutoModel.from_pretrained(model_name)
                   tokenizer = AutoTokenizer.from_pretrained(model_name)
                   ```
                """)
            
            log_user_action("model_download_completed", model_id=model['id'])
            
        except Exception as e:
            error(f"Error preparing download for {model['id']}: {str(e)}")
            st.error(f"❌ Error preparing download: {str(e)}")


def generate_download_script(model: Dict[str, Any]) -> str:
    """Generate a Python script for downloading and using the model."""
    script_template = f'''"""
Auto-generated download script for {model['name']}
Model ID: {model['id']}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os

def download_and_setup_model():
    """Download and setup the model with proper configuration."""
    
    model_name = "{model['id']}"
    print(f"Downloading {{model_name}}...")
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model with appropriate configuration
        print("Downloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model downloaded successfully!")
        print(f"Model: {{model_name}}")
        print(f"Device: {{'CUDA' if torch.cuda.is_available() else 'CPU'}}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error downloading model: {{e}}")
        return None, None

def main():
    """Main function to download and test the model."""
    model, tokenizer = download_and_setup_model()
    
    if model and tokenizer:
        print("\\n🎉 Model is ready to use!")
        print("\\nExample usage:")
        print("```python")
        print("from transformers import AutoModel, AutoTokenizer")
        print(f"model = AutoModel.from_pretrained('{model['id']}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{model['id']}')")
        print("```")
    else:
        print("❌ Failed to download model")

if __name__ == "__main__":
    main()
'''
    return script_template


def render_download_section():
    """Render the download management section."""
    st.markdown("## 📥 Download Management")
    
    download_dir = Path("output/downloaded_models")
    
    if download_dir.exists():
        downloaded_files = list(download_dir.glob("*.json"))
        
        if downloaded_files:
            st.markdown(f"### Downloaded Models ({len(downloaded_files)})")
            
            for file_path in downloaded_files:
                with open(file_path, 'r') as f:
                    model_info = json.load(f)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**{model_info['name']}** (`{model_info['id']}`)")
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{model_info['id']}"):
                        try:
                            file_path.unlink()
                            st.success("Model info removed!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error removing file: {e}")
                
                with col3:
                    if st.button("📋 Info", key=f"info_{model_info['id']}"):
                        st.json(model_info)
        else:
            st.info("No models downloaded yet.")
    else:
        st.info("Download directory not found. Download a model to get started.")
    
    # Download statistics
    st.markdown("### 📊 Download Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Categories", len(ModelCatalogManager().catalog_data))
    
    with col2:
        total_models = sum(
            sum(subcat['count'] for subcat in cat['categories'].values())
            for cat in ModelCatalogManager().catalog_data.values()
        )
        st.metric("Total Models", f"{total_models:,}")
    
    with col3:
        downloaded_count = len(list(download_dir.glob("*.json"))) if download_dir.exists() else 0
        st.metric("Downloaded Models", downloaded_count)


# Main execution
if __name__ == "__main__":
    render_model_catalog_page()
