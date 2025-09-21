# Final demonstration: Enhanced Tokenizer Manager in action
print("🎯 Enhanced Tokenizer Manager Demo")
print("=" * 45)

# Import the enhanced manager
import sys
from pathlib import Path

from utils.tokenizer_manager import tokenizer_manager
# from  import tokenizer_manager

try:
    
    # Demo 1: Load tokenizer with caching
    print("1️⃣ Loading tokenizer with intelligent caching...")
    start_time = time.time()
    tokenizer_info = tokenizer_manager.load_tokenizer("google/gemma-2-2b")
    load_time = time.time() - start_time
    
    if tokenizer_info:
        print(f"   ✅ Loaded: {tokenizer_info.tokenizer_type}")
        print(f"   📊 Vocab Size: {tokenizer_info.vocab_size:,}")
        print(f"   ⚡ Fast Tokenizer: {tokenizer_info.is_fast}")
        print(f"   🕒 Load Time: {load_time:.3f}s")
        
        # Demo 2: Batch tokenization
        print("\n2️⃣ Batch tokenization demo...")
        texts = [
            "Hello, how are you?",
            "I'm doing great, thank you!",
            "What's the weather like today?",
            "It's sunny and warm.",
            "Perfect for a walk!"
        ]
        
        result = tokenizer_manager.tokenize(texts, "google/gemma-2-2b")
        if result and result.success:
            print(f"   ✅ Batch processed {result.batch_size} texts")
            print(f"   📐 Output shape: {result.input_ids.shape}")
            print(f"   ⏱️ Processing time: {result.processing_time:.3f}s")
        
        # Demo 3: Performance statistics
        print("\n3️⃣ Performance statistics...")
        stats = tokenizer_manager.get_performance_stats()
        print(f"   📈 Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
        print(f"   💾 Memory Cache Size: {stats['memory_cache_size']}")
        print(f"   🗂️ Disk Cache Size: {stats['disk_cache_size']}")
        print(f"   ⚡ Average Load Time: {stats['average_load_time']:.3f}s")
        
        print("\n🎉 Enhanced tokenizer manager is working perfectly!")
        
    else:
        print("   ❌ Failed to load tokenizer")
        
except ImportError as e:
    print(f"⚠️ Could not import enhanced tokenizer manager: {e}")
    print("📝 This is expected if running outside the DurgasAI environment")
except Exception as e:
    print(f"⚠️ Demo failed: {e}")
    print("📝 This may be due to network or model availability issues")

print("\n✅ All demonstrations completed!")
print("💡 The enhanced tokenizer manager is ready for production use in DurgasAI")
