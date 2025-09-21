"""
Facial Feature Extractor for AI Image Generation.

This module provides comprehensive facial feature extraction capabilities using MediaPipe and OpenCV.
It integrates with the existing DurgasAI architecture to support facial analysis for AI image generation.

Key Features:
- MediaPipe-based facial landmark detection
- Facial attribute analysis (shape, expressions, etc.)
- Control map generation for generative models
- Integration with existing model management system
- Comprehensive error handling and logging

Key Classes:
- FacialFeatureExtractor: Main facial analysis interface
- FacialAttributeAnalyzer: Detailed facial attribute analysis
- ControlMapGenerator: Control map generation for AI models
"""

import os
import cv2
import numpy as np
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import time

# MediaPipe imports
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")

# Import existing configuration and logging
from .config import Config
from .logger import debug, info, warning, error, log_model_operation, time_operation, LoggedOperation


class FacialAttributeAnalyzer:
    """
    Analyzes facial attributes from landmark data.
    
    This class provides detailed analysis of facial features including:
    - Face shape classification
    - Eye shape and characteristics
    - Nose type and width
    - Mouth and lip analysis
    - Expression parameters
    """
    
    def __init__(self):
        """Initialize the facial attribute analyzer."""
        self.face_regions = {
            'face_oval': list(range(0, 17)) + list(range(17, 22)) + list(range(22, 27)) + 
                        list(range(27, 31)) + list(range(31, 36)),
            'left_eye': list(range(33, 42)) + list(range(159, 165)),
            'right_eye': list(range(362, 373)) + list(range(386, 392)),
            'nose': list(range(1, 5)) + list(range(6, 11)) + list(range(19, 25)),
            'mouth': list(range(61, 68)) + list(range(84, 91)) + list(range(267, 272)) + 
                    list(range(271, 276)),
            'eyebrows': list(range(70, 76)) + list(range(107, 113))
        }
        
        debug("FacialAttributeAnalyzer initialized", "facial_extraction")
    
    def analyze_face_shape(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """
        Analyze face shape from landmarks.
        
        Args:
            landmarks (List[Dict]): Facial landmarks
            
        Returns:
            Dict[str, Any]: Face shape analysis results
        """
        if not landmarks or len(landmarks) < 468:
            return {"error": "Insufficient landmarks for face shape analysis"}
        
        points = np.array([(lm["x"], lm["y"]) for lm in landmarks])
        
        # Calculate face dimensions
        face_width = self._calculate_distance(points[172], points[397])  # Left to right face
        face_height = self._calculate_distance(points[10], points[152])  # Top to bottom face
        jaw_width = self._calculate_distance(points[172], points[397])   # Jaw width
        forehead_width = self._calculate_distance(points[70], points[300])  # Forehead width
        
        # Calculate ratios
        face_ratio = face_height / face_width if face_width > 0 else 0
        jaw_to_forehead_ratio = jaw_width / forehead_width if forehead_width > 0 else 0
        
        # Classify face shape
        if face_ratio > 1.3:
            if jaw_to_forehead_ratio > 1.1:
                face_shape = "oval"
            else:
                face_shape = "long"
        elif face_ratio < 1.0:
            face_shape = "round"
        elif jaw_to_forehead_ratio > 1.1:
            face_shape = "heart"
        else:
            face_shape = "square"
        
        return {
            "face_shape": face_shape,
            "face_ratio": face_ratio,
            "jaw_to_forehead_ratio": jaw_to_forehead_ratio,
            "face_width": face_width,
            "face_height": face_height,
            "jaw_width": jaw_width,
            "forehead_width": forehead_width
        }
    
    def analyze_eye_features(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """
        Analyze eye features and characteristics.
        
        Args:
            landmarks (List[Dict]): Facial landmarks
            
        Returns:
            Dict[str, Any]: Eye analysis results
        """
        if not landmarks or len(landmarks) < 468:
            return {"error": "Insufficient landmarks for eye analysis"}
        
        points = np.array([(lm["x"], lm["y"]) for lm in landmarks])
        
        # Left eye analysis
        left_eye_points = [points[i] for i in range(33, 42)]
        left_eye_width = self._calculate_eye_width(left_eye_points)
        left_eye_height = self._calculate_eye_height(left_eye_points)
        
        # Right eye analysis
        right_eye_points = [points[i] for i in range(362, 373)]
        right_eye_width = self._calculate_eye_width(right_eye_points)
        right_eye_height = self._calculate_eye_height(right_eye_points)
        
        # Average measurements
        avg_eye_width = (left_eye_width + right_eye_width) / 2
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        eye_aspect_ratio = avg_eye_height / avg_eye_width if avg_eye_width > 0 else 0
        
        # Classify eye shape
        if eye_aspect_ratio > 0.4:
            eye_shape = "round"
        elif eye_aspect_ratio < 0.25:
            eye_shape = "narrow"
        else:
            eye_shape = "almond"
        
        return {
            "eye_shape": eye_shape,
            "eye_aspect_ratio": eye_aspect_ratio,
            "left_eye_width": left_eye_width,
            "right_eye_width": right_eye_width,
            "left_eye_height": left_eye_height,
            "right_eye_height": right_eye_height,
            "avg_eye_width": avg_eye_width,
            "avg_eye_height": avg_eye_height
        }
    
    def analyze_nose_features(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """
        Analyze nose features and characteristics.
        
        Args:
            landmarks (List[Dict]): Facial landmarks
            
        Returns:
            Dict[str, Any]: Nose analysis results
        """
        if not landmarks or len(landmarks) < 468:
            return {"error": "Insufficient landmarks for nose analysis"}
        
        points = np.array([(lm["x"], lm["y"]) for lm in landmarks])
        
        # Nose measurements
        nose_width = self._calculate_distance(points[31], points[35])
        nose_length = self._calculate_distance(points[1], points[5])
        nose_bridge_width = self._calculate_distance(points[6], points[8])
        
        # Nose type classification
        if nose_width < 30:
            nose_type = "narrow"
        elif nose_width > 45:
            nose_type = "wide"
        else:
            nose_type = "medium"
        
        return {
            "nose_type": nose_type,
            "nose_width": nose_width,
            "nose_length": nose_length,
            "nose_bridge_width": nose_bridge_width
        }
    
    def analyze_mouth_features(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """
        Analyze mouth and lip features.
        
        Args:
            landmarks (List[Dict]): Facial landmarks
            
        Returns:
            Dict[str, Any]: Mouth analysis results
        """
        if not landmarks or len(landmarks) < 468:
            return {"error": "Insufficient landmarks for mouth analysis"}
        
        points = np.array([(lm["x"], lm["y"]) for lm in landmarks])
        
        # Mouth measurements
        mouth_width = self._calculate_distance(points[61], points[291])
        upper_lip_height = self._calculate_distance(points[13], points[14])
        lower_lip_height = self._calculate_distance(points[17], points[18])
        
        # Lip fullness classification
        total_lip_height = upper_lip_height + lower_lip_height
        if total_lip_height > 20:
            lip_fullness = "full"
        elif total_lip_height < 10:
            lip_fullness = "thin"
        else:
            lip_fullness = "medium"
        
        return {
            "lip_fullness": lip_fullness,
            "mouth_width": mouth_width,
            "upper_lip_height": upper_lip_height,
            "lower_lip_height": lower_lip_height,
            "total_lip_height": total_lip_height
        }
    
    def _calculate_distance(self, point1: Tuple, point2: Tuple) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_eye_width(self, eye_points: List[Tuple]) -> float:
        """Calculate eye width from eye landmark points."""
        if len(eye_points) < 2:
            return 0
        return self._calculate_distance(eye_points[0], eye_points[-1])
    
    def _calculate_eye_height(self, eye_points: List[Tuple]) -> float:
        """Calculate eye height from eye landmark points."""
        if len(eye_points) < 4:
            return 0
        # Use top and bottom points for height
        top_point = eye_points[1]  # Upper eyelid
        bottom_point = eye_points[4]  # Lower eyelid
        return self._calculate_distance(top_point, bottom_point)


class ControlMapGenerator:
    """
    Generates control maps for generative AI models.
    
    This class creates various types of control maps from facial landmarks
    that can be used to guide generative models like Stable Diffusion with ControlNet.
    """
    
    def __init__(self):
        """Initialize the control map generator."""
        debug("ControlMapGenerator initialized", "facial_extraction")
    
    def generate_landmark_control_map(self, landmarks: List[Dict], 
                                    image_dimensions: Dict, 
                                    map_type: str = "points") -> np.ndarray:
        """
        Generate control map from facial landmarks.
        
        Args:
            landmarks (List[Dict]): Facial landmarks
            image_dimensions (Dict): Image width and height
            map_type (str): Type of control map ("points", "contours", "heatmap")
            
        Returns:
            np.ndarray: Control map image
        """
        width, height = image_dimensions["width"], image_dimensions["height"]
        control_map = np.zeros((height, width), dtype=np.uint8)
        
        if map_type == "points":
            return self._generate_points_map(landmarks, control_map)
        elif map_type == "contours":
            return self._generate_contours_map(landmarks, control_map)
        elif map_type == "heatmap":
            return self._generate_heatmap(landmarks, control_map)
        else:
            return self._generate_points_map(landmarks, control_map)
    
    def _generate_points_map(self, landmarks: List[Dict], control_map: np.ndarray) -> np.ndarray:
        """Generate control map with landmark points."""
        for landmark in landmarks:
            x, y = int(landmark["x"]), int(landmark["y"])
            if 0 <= x < control_map.shape[1] and 0 <= y < control_map.shape[0]:
                cv2.circle(control_map, (x, y), 2, 255, -1)
        return control_map
    
    def _generate_contours_map(self, landmarks: List[Dict], control_map: np.ndarray) -> np.ndarray:
        """Generate control map with facial contours."""
        # Define facial contour points
        face_oval_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        
        # Draw face outline
        for i in range(len(face_oval_indices) - 1):
            idx1 = face_oval_indices[i]
            idx2 = face_oval_indices[i + 1]
            if idx1 < len(landmarks) and idx2 < len(landmarks):
                pt1 = (int(landmarks[idx1]["x"]), int(landmarks[idx1]["y"]))
                pt2 = (int(landmarks[idx2]["x"]), int(landmarks[idx2]["y"]))
                cv2.line(control_map, pt1, pt2, 200, 2)
        
        return control_map
    
    def _generate_heatmap(self, landmarks: List[Dict], control_map: np.ndarray) -> np.ndarray:
        """Generate control map as a heatmap."""
        height, width = control_map.shape
        
        for landmark in landmarks:
            x, y = int(landmark["x"]), int(landmark["y"])
            if 0 <= x < width and 0 <= y < height:
                # Create a small Gaussian-like heat spot
                cv2.circle(control_map, (x, y), 5, 100, -1)
        
        # Apply Gaussian blur for heatmap effect
        control_map = cv2.GaussianBlur(control_map, (15, 15), 0)
        
        return control_map


class FacialFeatureExtractor:
    """
    Main facial feature extraction interface.
    
    This class provides a comprehensive interface for facial feature extraction
    that integrates with the existing DurgasAI architecture.
    """
    
    def __init__(self):
        """Initialize the facial feature extractor."""
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for facial feature extraction. Install with: pip install mediapipe")
        
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize analysis components
        self.attribute_analyzer = FacialAttributeAnalyzer()
        self.control_map_generator = ControlMapGenerator()
        
        # Performance tracking
        self.total_extractions = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        
        info("FacialFeatureExtractor initialized successfully", "facial_extraction")
        log_model_operation("facial_extractor_initialized")
    
    @time_operation("facial_feature_extraction", "facial_extraction")
    def extract_features(self, image_path: str) -> Dict[str, Any]:
        """
        Complete facial feature extraction pipeline.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Dict[str, Any]: Complete feature extraction results
        """
        start_time = time.time()
        self.total_extractions += 1
        
        debug(f"Starting facial feature extraction", "facial_extraction",
              image_path=image_path[:50] + "..." if len(image_path) > 50 else image_path)
        
        try:
            with LoggedOperation("facial_feature_extraction", "facial_extraction",
                               extra_data={"image_path": image_path}):
                
                # Step 1: Detect landmarks
                landmark_result = self._detect_landmarks(image_path)
                if not landmark_result["success"]:
                    self.failed_extractions += 1
                    return landmark_result
                
                landmarks = landmark_result["landmarks"]
                image_dimensions = landmark_result["image_dimensions"]
                
                # Step 2: Analyze facial attributes
                attributes = self._analyze_facial_attributes(landmarks)
                
                # Step 3: Generate control maps
                control_maps = self._generate_control_maps(landmarks, image_dimensions)
                
                # Step 4: Compile results
                processing_time = time.time() - start_time
                self.successful_extractions += 1
                
                result = {
                    "success": True,
                    "landmarks": landmarks,
                    "attributes": attributes,
                    "control_maps": control_maps,
                    "image_dimensions": image_dimensions,
                    "metadata": {
                        "landmark_count": len(landmarks),
                        "processing_time": processing_time,
                        "extraction_method": "MediaPipe",
                        "face_count": landmark_result.get("face_count", 1)
                    }
                }
                
                info(f"Facial feature extraction completed successfully in {processing_time:.2f}s", 
                     "facial_extraction",
                     landmark_count=len(landmarks),
                     processing_time=processing_time)
                
                log_model_operation("facial_extraction_completed",
                                  processing_time=processing_time,
                                  success=True)
                
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.failed_extractions += 1
            
            error_msg = f"Facial feature extraction failed: {str(e)}"
            error(error_msg, "facial_extraction", e,
                  processing_time=processing_time)
            
            log_model_operation("facial_extraction_failed",
                              error=str(e),
                              processing_time=processing_time)
            
            return {
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }
    
    def _detect_landmarks(self, image_path: str) -> Dict[str, Any]:
        """
        Detect facial landmarks from an image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            Dict[str, Any]: Landmark detection results
        """
        # Validate image file
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image file not found: {image_path}"}
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": f"Could not load image: {image_path}"}
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image_rgb.shape
        
        # Process image with MediaPipe
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return {"success": False, "error": "No face detected in image"}
        
        # Extract landmarks
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            landmarks.append({"x": x, "y": y, "z": z})
        
        return {
            "success": True,
            "landmarks": landmarks,
            "image_dimensions": {"width": width, "height": height},
            "face_count": len(results.multi_face_landmarks)
        }
    
    def _analyze_facial_attributes(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """
        Analyze facial attributes from landmarks.
        
        Args:
            landmarks (List[Dict]): Facial landmarks
            
        Returns:
            Dict[str, Any]: Facial attribute analysis
        """
        try:
            face_shape = self.attribute_analyzer.analyze_face_shape(landmarks)
            eye_features = self.attribute_analyzer.analyze_eye_features(landmarks)
            nose_features = self.attribute_analyzer.analyze_nose_features(landmarks)
            mouth_features = self.attribute_analyzer.analyze_mouth_features(landmarks)
            
            return {
                "face_shape": face_shape,
                "eye_features": eye_features,
                "nose_features": nose_features,
                "mouth_features": mouth_features
            }
            
        except Exception as e:
            error(f"Facial attribute analysis failed: {str(e)}", "facial_extraction", e)
            return {"error": f"Attribute analysis failed: {str(e)}"}
    
    def _generate_control_maps(self, landmarks: List[Dict], 
                             image_dimensions: Dict) -> Dict[str, np.ndarray]:
        """
        Generate various control maps for generative models.
        
        Args:
            landmarks (List[Dict]): Facial landmarks
            image_dimensions (Dict): Image dimensions
            
        Returns:
            Dict[str, np.ndarray]: Different types of control maps
        """
        try:
            control_maps = {}
            
            # Generate different types of control maps
            control_maps["points"] = self.control_map_generator.generate_landmark_control_map(
                landmarks, image_dimensions, "points"
            )
            control_maps["contours"] = self.control_map_generator.generate_landmark_control_map(
                landmarks, image_dimensions, "contours"
            )
            control_maps["heatmap"] = self.control_map_generator.generate_landmark_control_map(
                landmarks, image_dimensions, "heatmap"
            )
            
            return control_maps
            
        except Exception as e:
            error(f"Control map generation failed: {str(e)}", "facial_extraction", e)
            return {"error": f"Control map generation failed: {str(e)}"}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the facial feature extractor.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        success_rate = (self.successful_extractions / self.total_extractions * 100) if self.total_extractions > 0 else 0
        
        return {
            "total_extractions": self.total_extractions,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "success_rate": success_rate
        }
    
    def save_features_to_file(self, features: Dict[str, Any], output_path: str) -> bool:
        """
        Save extracted features to a JSON file.
        
        Args:
            features (Dict[str, Any]): Extracted features
            output_path (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_features = self._make_features_serializable(features)
            
            with open(output_path, 'w') as f:
                json.dump(serializable_features, f, indent=2)
            
            info(f"Features saved to {output_path}", "facial_extraction")
            return True
            
        except Exception as e:
            error(f"Failed to save features to file: {str(e)}", "facial_extraction", e)
            return False
    
    def _make_features_serializable(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert features to JSON-serializable format.
        
        Args:
            features (Dict[str, Any]): Features to serialize
            
        Returns:
            Dict[str, Any]: Serializable features
        """
        serializable = {}
        
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = self._make_features_serializable(value)
            else:
                serializable[key] = value
        
        return serializable


# Factory function for easy integration
def create_facial_extractor() -> Optional[FacialFeatureExtractor]:
    """
    Factory function to create FacialFeatureExtractor instance.
    
    Returns:
        Optional[FacialFeatureExtractor]: Facial feature extractor instance or None if failed
    """
    try:
        if not MEDIAPIPE_AVAILABLE:
            error("MediaPipe not available for facial feature extraction", "facial_extraction")
            return None
        
        return FacialFeatureExtractor()
        
    except Exception as e:
        error(f"Failed to create facial feature extractor: {str(e)}", "facial_extraction", e)
        return None
