"""
TensorFlow and ML library warning suppression module.
Simplified version to avoid import and stderr issues.
"""

import os
import warnings
import logging

# Set environment variables before any TensorFlow imports
# Try to use centralized config system, fallback to direct setting
try:
    import sys
    from pathlib import Path
    
    # Add utils to path
    utils_path = Path(__file__).parent / 'utils'
    if utils_path.exists():
        sys.path.insert(0, str(utils_path.parent))
        from utils.config import Config
        
        # Setup environment variables from configuration
        Config.setup_environment_variables()
        print("✅ Environment variables loaded from centralized config")
    else:
        raise ImportError("Utils path not found")
        
except Exception as e:
    # Fallback to direct environment variable setting
    print(f"⚠️ Could not load centralized config ({e}), using fallback")
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF2_BEHAVIOR', '1')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('PYTHONWARNINGS', 'ignore')

# Suppress Python warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Specific TensorFlow/Keras warnings (use raw strings to fix regex escape sequences)
warnings.filterwarnings('ignore', message=r'.*deprecated.*')
warnings.filterwarnings('ignore', message=r'.*tf\..*')
warnings.filterwarnings('ignore', message=r'.*reset_default_graph.*')
warnings.filterwarnings('ignore', message=r'.*oneDNN.*')
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='keras')
warnings.filterwarnings('ignore', module='transformers')

# Suppress logging from ML libraries
try:
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('keras').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
except Exception:
    # Ignore any errors during logger setup
    pass

# Simple TensorFlow initialization without stderr redirection
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except ImportError:
    # TensorFlow not available, which is fine
    pass
except Exception:
    # Any other error during TensorFlow import, ignore
    pass
