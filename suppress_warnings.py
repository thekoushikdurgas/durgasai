"""
TensorFlow and ML library warning suppression module.
Import this module at the very beginning to suppress all ML-related warnings.
"""

import os
import warnings
import logging
import sys

def suppress_tf_warnings():
    """Comprehensive TensorFlow warning suppression."""
    
    # Set environment variables before any TensorFlow imports
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Specific TensorFlow/Keras warnings
    warnings.filterwarnings('ignore', message='.*deprecated.*')
    warnings.filterwarnings('ignore', message='.*tf\..*')
    warnings.filterwarnings('ignore', message='.*reset_default_graph.*')
    warnings.filterwarnings('ignore', message='.*oneDNN.*')
    warnings.filterwarnings('ignore', module='tensorflow')
    warnings.filterwarnings('ignore', module='keras')
    warnings.filterwarnings('ignore', module='transformers')
    
    # Suppress logging from ML libraries
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('keras').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    
    # Redirect stderr temporarily to suppress C++ warnings
    class NullWriter:
        def write(self, txt): pass
        def flush(self): pass
    
    # Store original stderr
    original_stderr = sys.stderr
    
    # Temporarily redirect stderr during imports
    sys.stderr = NullWriter()
    
    try:
        # Import TensorFlow to trigger any remaining warnings
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except ImportError:
        pass
    finally:
        # Restore stderr
        sys.stderr = original_stderr

# Call the suppression function when this module is imported
suppress_tf_warnings()
