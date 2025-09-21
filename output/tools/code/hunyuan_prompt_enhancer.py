import re
import random
from typing import Dict, Any, List, Optional

def enhance_hunyuan_prompt(
    prompt: str,
    enhancement_type: str = "comprehensive",
    style_preference: str = "balanced",
    target_language: str = "english",
    quality_level: str = "high",
    add_technical_tags: bool = True,
    add_lighting_descriptions: bool = True,
    max_length: int = 200
) -> Dict[str, Any]:
    """
    Enhance and optimize prompts for HunyuanImage-2.1 generation.
    
    Args:
        prompt (str): Original prompt to enhance
        enhancement_type (str): Type of enhancement to apply
        style_preference (str): Artistic style preference
        target_language (str): Target language for multilingual enhancement
        quality_level (str): Quality level for enhancement
        add_technical_tags (bool): Add technical quality tags
        add_lighting_descriptions (bool): Add lighting and atmosphere descriptions
        max_length (int): Maximum length for enhanced prompt
        
    Returns:
        Dict[str, Any]: Enhanced prompt and enhancement details
    """
    
    try:
        original_prompt = prompt.strip()
        
        if not original_prompt:
            return {
                "success": False,
                "error": "Empty prompt provided"
            }
        
        # Initialize enhancement components
        enhancements = []
        improvements = []
        additions = []
        
        # Base enhancement based on type
        if enhancement_type == "comprehensive":
            enhanced_prompt = _comprehensive_enhancement(
                original_prompt, style_preference, quality_level
            )
        elif enhancement_type == "style":
            enhanced_prompt = _style_enhancement(
                original_prompt, style_preference
            )
        elif enhancement_type == "detail":
            enhanced_prompt = _detail_enhancement(
                original_prompt, quality_level
            )
        elif enhancement_type == "quality":
            enhanced_prompt = _quality_enhancement(
                original_prompt, quality_level
            )
        elif enhancement_type == "multilingual":
            enhanced_prompt = _multilingual_enhancement(
                original_prompt, target_language
            )
        elif enhancement_type == "custom":
            enhanced_prompt = _custom_enhancement(
                original_prompt, style_preference, quality_level
            )
        else:
            enhanced_prompt = _comprehensive_enhancement(
                original_prompt, style_preference, quality_level
            )
        
        # Add technical tags if requested
        if add_technical_tags:
            technical_tags = _get_technical_tags(quality_level)
            enhanced_prompt = f"{enhanced_prompt}, {technical_tags}"
            additions.append("Technical quality tags")
        
        # Add lighting descriptions if requested
        if add_lighting_descriptions:
            lighting_desc = _get_lighting_description(style_preference)
            enhanced_prompt = f"{enhanced_prompt}, {lighting_desc}"
            additions.append("Lighting and atmosphere descriptions")
        
        # Ensure prompt doesn't exceed max length
        if len(enhanced_prompt) > max_length:
            enhanced_prompt = _truncate_prompt(enhanced_prompt, max_length)
            improvements.append("Truncated to fit length limit")
        
        # Calculate improvements
        length_increase = len(enhanced_prompt) - len(original_prompt)
        improvements.extend([
            f"Enhanced from {len(original_prompt)} to {len(enhanced_prompt)} characters",
            f"Applied {enhancement_type} enhancement",
            f"Optimized for {style_preference} style"
        ])
        
        # Calculate confidence score
        confidence_score = _calculate_confidence_score(enhanced_prompt, quality_level)
        
        # Generate alternative prompts
        alternative_prompts = _generate_alternative_prompts(
            original_prompt, style_preference, quality_level
        )
        
        # Generate recommendations
        recommendations = _generate_recommendations(enhanced_prompt, style_preference)
        
        return {
            "success": True,
            "enhanced_prompt": enhanced_prompt,
            "original_prompt": original_prompt,
            "enhancement_details": {
                "enhancement_type": enhancement_type,
                "style_preference": style_preference,
                "quality_level": quality_level,
                "additions": additions,
                "improvements": improvements,
                "length_increase": length_increase,
                "confidence_score": confidence_score
            },
            "alternative_prompts": alternative_prompts,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error enhancing prompt: {str(e)}"
        }

def _comprehensive_enhancement(prompt: str, style: str, quality: str) -> str:
    """Apply comprehensive enhancement to prompt."""
    
    # Style-specific enhancements
    style_modifiers = {
        "professional": "professional, clean, polished, high-quality",
        "artistic": "artistic, creative, expressive, dramatic",
        "casual": "casual, relaxed, natural, approachable",
        "fantasy": "fantasy, magical, mystical, ethereal",
        "vintage": "vintage, classic, timeless, nostalgic",
        "photorealistic": "photorealistic, detailed, lifelike, high-resolution",
        "balanced": "balanced, well-composed, harmonious"
    }
    
    # Quality-based enhancements
    quality_modifiers = {
        "basic": "good quality",
        "medium": "high quality, detailed",
        "high": "high quality, detailed, sharp",
        "premium": "ultra high quality, extremely detailed, sharp focus"
    }
    
    # Extract and enhance subject
    enhanced = prompt
    
    # Add style modifiers
    if style in style_modifiers:
        enhanced = f"{enhanced}, {style_modifiers[style]}"
    
    # Add quality modifiers
    if quality in quality_modifiers:
        enhanced = f"{enhanced}, {quality_modifiers[quality]}"
    
    # Add composition improvements
    composition_tags = _get_composition_tags(style)
    enhanced = f"{enhanced}, {composition_tags}"
    
    return enhanced

def _style_enhancement(prompt: str, style: str) -> str:
    """Apply style-specific enhancement."""
    
    style_templates = {
        "professional": f"{prompt}, professional photography, clean background, studio lighting, business attire",
        "artistic": f"{prompt}, artistic interpretation, creative composition, expressive style, artistic lighting",
        "casual": f"{prompt}, casual setting, natural pose, relaxed atmosphere, everyday clothing",
        "fantasy": f"{prompt}, fantasy setting, magical atmosphere, ethereal lighting, mystical elements",
        "vintage": f"{prompt}, vintage style, classic composition, retro aesthetics, nostalgic atmosphere",
        "photorealistic": f"{prompt}, photorealistic, detailed, lifelike, high-resolution photography"
    }
    
    return style_templates.get(style, f"{prompt}, {style} style")

def _detail_enhancement(prompt: str, quality: str) -> str:
    """Add detailed descriptions to prompt."""
    
    detail_levels = {
        "basic": "with basic details",
        "medium": "with detailed features and textures",
        "high": "with intricate details, fine textures, and subtle nuances",
        "premium": "with extremely intricate details, fine textures, subtle nuances, and perfect clarity"
    }
    
    detail_desc = detail_levels.get(quality, "with detailed features")
    return f"{prompt}, {detail_desc}"

def _quality_enhancement(prompt: str, quality: str) -> str:
    """Enhance prompt with quality descriptors."""
    
    quality_descriptors = {
        "basic": "good quality",
        "medium": "high quality, well-detailed",
        "high": "high quality, detailed, sharp focus",
        "premium": "ultra high quality, extremely detailed, sharp focus, professional grade"
    }
    
    quality_desc = quality_descriptors.get(quality, "high quality")
    return f"{prompt}, {quality_desc}"

def _multilingual_enhancement(prompt: str, target_language: str) -> str:
    """Create multilingual version of prompt."""
    
    if target_language == "chinese":
        # Simple translation for common terms (in practice, would use proper translation)
        chinese_terms = {
            "beautiful": "美丽的",
            "landscape": "风景",
            "portrait": "肖像",
            "professional": "专业的",
            "artistic": "艺术的",
            "high quality": "高质量",
            "detailed": "详细的"
        }
        
        enhanced = prompt
        for eng, chn in chinese_terms.items():
            enhanced = enhanced.replace(eng, f"{eng} ({chn})")
        
        return enhanced
    
    elif target_language == "both":
        # Add both English and Chinese descriptors
        return f"{prompt}, high quality (高质量), detailed (详细的)"
    
    else:
        # English only
        return prompt

def _custom_enhancement(prompt: str, style: str, quality: str) -> str:
    """Apply custom enhancement based on prompt analysis."""
    
    # Analyze prompt content
    has_person = any(word in prompt.lower() for word in ['person', 'man', 'woman', 'people', 'portrait'])
    has_scene = any(word in prompt.lower() for word in ['landscape', 'city', 'garden', 'room', 'building'])
    has_object = any(word in prompt.lower() for word in ['car', 'animal', 'flower', 'tree', 'house'])
    
    enhanced = prompt
    
    # Add context-appropriate enhancements
    if has_person:
        enhanced = f"{enhanced}, well-lit, good composition"
    elif has_scene:
        enhanced = f"{enhanced}, atmospheric, well-composed"
    elif has_object:
        enhanced = f"{enhanced}, detailed, well-positioned"
    
    # Add style and quality
    enhanced = _comprehensive_enhancement(enhanced, style, quality)
    
    return enhanced

def _get_technical_tags(quality: str) -> str:
    """Get technical quality tags based on quality level."""
    
    technical_tags = {
        "basic": "good composition",
        "medium": "good composition, proper lighting",
        "high": "good composition, proper lighting, sharp details",
        "premium": "perfect composition, professional lighting, ultra-sharp details, masterful execution"
    }
    
    return technical_tags.get(quality, "good composition")

def _get_lighting_description(style: str) -> str:
    """Get lighting description based on style."""
    
    lighting_descriptions = {
        "professional": "professional studio lighting",
        "artistic": "artistic lighting with dramatic shadows",
        "casual": "natural lighting, soft illumination",
        "fantasy": "ethereal lighting, magical glow",
        "vintage": "classic lighting, warm tones",
        "photorealistic": "natural lighting, realistic illumination",
        "balanced": "balanced lighting, good contrast"
    }
    
    return lighting_descriptions.get(style, "good lighting")

def _get_composition_tags(style: str) -> str:
    """Get composition tags based on style."""
    
    composition_tags = {
        "professional": "well-composed, centered",
        "artistic": "creative composition, dynamic framing",
        "casual": "natural composition, relaxed framing",
        "fantasy": "dramatic composition, mystical framing",
        "vintage": "classic composition, timeless framing",
        "photorealistic": "realistic composition, natural framing",
        "balanced": "balanced composition, harmonious framing"
    }
    
    return composition_tags.get(style, "well-composed")

def _truncate_prompt(prompt: str, max_length: int) -> str:
    """Truncate prompt to fit within max length."""
    
    if len(prompt) <= max_length:
        return prompt
    
    # Try to truncate at word boundaries
    words = prompt.split(', ')
    truncated = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 2 <= max_length:  # +2 for ", "
            truncated.append(word)
            current_length += len(word) + 2
        else:
            break
    
    return ', '.join(truncated)

def _calculate_confidence_score(prompt: str, quality: str) -> float:
    """Calculate confidence score for the enhanced prompt."""
    
    base_score = 0.7
    
    # Quality-based adjustments
    quality_bonus = {
        "basic": 0.0,
        "medium": 0.1,
        "high": 0.15,
        "premium": 0.2
    }
    
    # Length-based adjustments
    length_bonus = min(len(prompt) / 200, 0.1)
    
    # Complexity-based adjustments
    complexity_bonus = min(prompt.count(',') * 0.02, 0.1)
    
    score = base_score + quality_bonus.get(quality, 0.1) + length_bonus + complexity_bonus
    
    return min(score, 1.0)

def _generate_alternative_prompts(original: str, style: str, quality: str) -> List[Dict[str, str]]:
    """Generate alternative enhanced prompts."""
    
    alternatives = []
    
    # Alternative 1: Minimal enhancement
    alternatives.append({
        "variant": "minimal",
        "prompt": f"{original}, high quality",
        "focus": "Simple quality enhancement"
    })
    
    # Alternative 2: Style-focused
    style_alt = _style_enhancement(original, style)
    alternatives.append({
        "variant": "style-focused",
        "prompt": style_alt,
        "focus": f"Emphasis on {style} style"
    })
    
    # Alternative 3: Technical
    tech_alt = f"{original}, ultra high quality, detailed, sharp focus, professional"
    alternatives.append({
        "variant": "technical",
        "prompt": tech_alt,
        "focus": "Technical quality emphasis"
    })
    
    return alternatives

def _generate_recommendations(prompt: str, style: str) -> Dict[str, Any]:
    """Generate recommendations for optimal generation."""
    
    # Determine aspect ratio based on content
    if any(word in prompt.lower() for word in ['portrait', 'person', 'face', 'head']):
        suggested_aspect_ratio = "3:4"
    elif any(word in prompt.lower() for word in ['landscape', 'city', 'scene', 'view']):
        suggested_aspect_ratio = "16:9"
    else:
        suggested_aspect_ratio = "1:1"
    
    # Determine inference steps based on quality
    if "ultra high quality" in prompt or "extremely detailed" in prompt:
        suggested_steps = 50
    else:
        suggested_steps = 25
    
    # Determine guidance scale
    if len(prompt) > 150:
        suggested_guidance = 4.0
    else:
        suggested_guidance = 3.5
    
    return {
        "suggested_aspect_ratio": suggested_aspect_ratio,
        "suggested_inference_steps": suggested_steps,
        "suggested_guidance_scale": suggested_guidance,
        "use_refiner": True,
        "use_prompt_enhancement": False  # Already enhanced
    }
