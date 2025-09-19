"""
Environment setup script for DurgasAI.
Sets up TensorFlow environment variables to suppress warnings.
"""

import os
import sys


def setup_tensorflow_env():
    """Setup TensorFlow environment variables to suppress warnings."""
    # Suppress TensorFlow warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Suppress Python warnings
    import warnings
    import logging
    
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*deprecated.*')
    warnings.filterwarnings('ignore', message='.*tf\..*')
    warnings.filterwarnings('ignore', message='.*reset_default_graph.*')
    
    # Suppress logging from ML libraries
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('keras').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    
    print("✅ TensorFlow environment variables configured:")
    print(f"   TF_ENABLE_ONEDNN_OPTS = {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
    print(f"   TF_CPP_MIN_LOG_LEVEL = {os.environ.get('TF_CPP_MIN_LOG_LEVEL')}")
    print(f"   TF2_BEHAVIOR = {os.environ.get('TF2_BEHAVIOR')}")
    print(f"   CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("   Python warnings and logging suppressed for TensorFlow/Keras")


def setup_huggingface_cache():
    """Setup HuggingFace cache directories."""
    cache_dir = os.path.join(os.path.dirname(__file__), "output", "cache")
    
    # Create cache directories if they don't exist
    hf_cache = os.path.join(cache_dir, "huggingface")
    transformers_cache = os.path.join(cache_dir, "transformers")
    
    os.makedirs(hf_cache, exist_ok=True)
    os.makedirs(transformers_cache, exist_ok=True)
    
    # Set cache environment variables
    os.environ['HF_HOME'] = hf_cache
    os.environ['TRANSFORMERS_CACHE'] = transformers_cache
    
    print("✅ HuggingFace cache directories configured:")
    print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
    print(f"   TRANSFORMERS_CACHE = {os.environ.get('TRANSFORMERS_CACHE')}")


def main():
    """Main setup function."""
    print("🚀 Setting up DurgasAI environment...")
    print()
    
    setup_tensorflow_env()
    print()
    
    setup_huggingface_cache()
    print()
    
    print("🎉 Environment setup complete!")
    print("You can now run the application with: streamlit run app.py")


if __name__ == "__main__":
    main()
