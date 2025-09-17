"""
Comprehensive Pydantic Models for Structured Outputs

This module provides a comprehensive collection of Pydantic models for various
structured output use cases, based on the official Ollama structured output examples
and extended for real-world applications.

Models are organized by category:
- Basic Examples (based on structured-outputs.py)
- Image Analysis (based on structured-outputs-image.py) 
- Extended Models for various domains
- Dynamic and utility models

Each model includes proper field validation, descriptions, and type hints
for optimal structured output generation with Ollama.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Literal, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import re
import uuid


# ==============================================================================
# BASIC MODELS (based on structured-outputs.py example)
# ==============================================================================

class PersonInfo(BaseModel):
    """Basic person information - simple structured output example"""
    name: str = Field(description="Full name of the person")
    age: int = Field(ge=0, le=150, description="Age in years")
    is_available: bool = Field(description="Whether the person is available")


class FriendInfo(BaseModel):
    """Friend information from the original structured-outputs.py example"""
    name: str = Field(description="Friend's name")
    age: int = Field(ge=0, le=150, description="Friend's age in years")
    is_available: bool = Field(description="Whether the friend is available to hang out")


class FriendList(BaseModel):
    """List of friends from the original structured-outputs.py example"""
    friends: List[FriendInfo] = Field(description="List of friends with their information")


# ==============================================================================
# IMAGE ANALYSIS MODELS (based on structured-outputs-image.py example)
# ==============================================================================

class DetectedObject(BaseModel):
    """Individual object detected in an image"""
    name: str = Field(description="Name/type of the detected object")
    confidence: float = Field(ge=0, le=1, description="Confidence score for object detection")
    attributes: str = Field(description="Additional attributes or characteristics of the object")


class ImageDescription(BaseModel):
    """Comprehensive image analysis from structured-outputs-image.py example"""
    summary: str = Field(description="Overall summary of the image content")
    objects: List[DetectedObject] = Field(description="List of objects detected in the image")
    scene: str = Field(description="Description of the overall scene or setting")
    colors: List[str] = Field(description="Prominent colors in the image")
    time_of_day: Literal['Morning', 'Afternoon', 'Evening', 'Night'] = Field(description="Time of day depicted")
    setting: Literal['Indoor', 'Outdoor', 'Unknown'] = Field(description="Whether the scene is indoor or outdoor")
    text_content: Optional[str] = Field(None, description="Any text content detected in the image")


# ==============================================================================
# EXTENDED STRUCTURED OUTPUT MODELS
# ==============================================================================

class Sentiment(str, Enum):
    """Sentiment classification enum"""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentAnalysis(BaseModel):
    """Sentiment analysis structured output"""
    text: str = Field(description="The original text analyzed")
    sentiment: Sentiment = Field(description="Overall sentiment classification")
    confidence: float = Field(ge=0, le=1, description="Confidence score for sentiment prediction")
    emotions: List[str] = Field(description="Specific emotions detected")
    key_phrases: List[str] = Field(description="Important phrases that influenced the sentiment")
    summary: str = Field(description="Brief summary of the sentiment analysis")


class CodeAnalysis(BaseModel):
    """Code analysis and review structured output"""
    language: str = Field(description="Programming language detected")
    code_type: Literal['function', 'class', 'script', 'snippet', 'configuration'] = Field(
        description="Type of code being analyzed"
    )
    complexity: Literal['Low', 'Medium', 'High'] = Field(description="Code complexity assessment")
    issues: List[str] = Field(default=[], description="Potential issues or bugs identified")
    suggestions: List[str] = Field(default=[], description="Improvement suggestions")
    dependencies: List[str] = Field(default=[], description="External dependencies identified")
    summary: str = Field(description="Overall analysis summary")
    maintainability_score: float = Field(ge=0, le=10, description="Maintainability score out of 10")


class DocumentSummary(BaseModel):
    """Document summarization structured output"""
    title: str = Field(description="Document title or main topic")
    summary: str = Field(description="Comprehensive summary of the document")
    key_points: List[str] = Field(description="Main key points or findings")
    topics: List[str] = Field(description="Major topics covered")
    word_count: Optional[int] = Field(None, description="Estimated word count")
    reading_time: Optional[str] = Field(None, description="Estimated reading time")
    difficulty: Literal['Beginner', 'Intermediate', 'Advanced', 'Expert'] = Field(
        description="Content difficulty level"
    )
    audience: str = Field(description="Target audience for this document")


class QuestionAnswer(BaseModel):
    """Question answering structured output"""
    question: str = Field(description="The original question asked")
    answer: str = Field(description="Comprehensive answer to the question")
    confidence: float = Field(ge=0, le=1, description="Confidence in the answer accuracy")
    sources: List[str] = Field(default=[], description="Sources or references used")
    related_questions: List[str] = Field(default=[], description="Related questions that might be asked")
    follow_up_suggestions: List[str] = Field(default=[], description="Suggested follow-up questions")
    answer_type: Literal['factual', 'opinion', 'explanation', 'instruction'] = Field(
        description="Type of answer provided"
    )


class TranslationResult(BaseModel):
    """Translation structured output"""
    original_text: str = Field(description="Original text to be translated")
    translated_text: str = Field(description="Translated text")
    source_language: str = Field(description="Detected or specified source language")
    target_language: str = Field(description="Target language for translation")
    confidence: float = Field(ge=0, le=1, description="Translation confidence score")
    alternative_translations: List[str] = Field(default=[], description="Alternative translation options")
    detected_entities: List[str] = Field(default=[], description="Named entities or special terms identified")
    cultural_notes: Optional[str] = Field(None, description="Cultural context or notes about the translation")


class DataExtractionResult(BaseModel):
    """Data extraction from unstructured text"""
    extracted_data: Dict[str, Any] = Field(description="Key-value pairs of extracted information")
    entities: List[str] = Field(description="Named entities found in the text")
    dates: List[str] = Field(default=[], description="Dates mentioned in the text")
    numbers: List[float] = Field(default=[], description="Numerical values extracted")
    locations: List[str] = Field(default=[], description="Geographic locations mentioned")
    organizations: List[str] = Field(default=[], description="Organizations or companies mentioned")
    people: List[str] = Field(default=[], description="People or person names mentioned")
    confidence: float = Field(ge=0, le=1, description="Overall extraction confidence")


class ProductReview(BaseModel):
    """Product review analysis structured output"""
    product_name: str = Field(description="Name of the product being reviewed")
    rating: float = Field(ge=1, le=5, description="Overall rating score (1-5)")
    sentiment: Sentiment = Field(description="Overall sentiment of the review")
    pros: List[str] = Field(description="Positive aspects mentioned")
    cons: List[str] = Field(description="Negative aspects mentioned")
    key_features: List[str] = Field(description="Key features discussed")
    recommendation: Literal['Highly Recommended', 'Recommended', 'Neutral', 'Not Recommended'] = Field(
        description="Overall recommendation"
    )
    summary: str = Field(description="Brief summary of the review")
    helpful_votes: Optional[int] = Field(None, description="Number of helpful votes if available")


class MeetingNotes(BaseModel):
    """Meeting notes structured output"""
    meeting_title: str = Field(description="Title or subject of the meeting")
    date: Optional[str] = Field(None, description="Meeting date if mentioned")
    attendees: List[str] = Field(description="Meeting participants")
    key_discussions: List[str] = Field(description="Main discussion points")
    decisions_made: List[str] = Field(description="Decisions or conclusions reached")
    action_items: List[str] = Field(description="Action items and tasks assigned")
    next_steps: List[str] = Field(description="Planned next steps or follow-ups")
    meeting_duration: Optional[str] = Field(None, description="Meeting duration if mentioned")
    priority_level: Literal['Low', 'Medium', 'High', 'Critical'] = Field(description="Priority level of the meeting content")


# ==============================================================================
# MULTIMODAL STRUCTURED MODELS (combining images with structured output)
# ==============================================================================

class VisualContentAnalysis(BaseModel):
    """Advanced image content analysis"""
    content_type: Literal['photograph', 'illustration', 'diagram', 'chart', 'screenshot', 'artwork'] = Field(
        description="Type of visual content"
    )
    main_subject: str = Field(description="Primary subject or focus of the image")
    description: str = Field(description="Detailed description of the image content")
    objects: List[DetectedObject] = Field(description="Objects detected in the image")
    colors: List[str] = Field(description="Dominant colors in the image")
    style: str = Field(description="Artistic or photographic style")
    quality_assessment: Literal['Poor', 'Fair', 'Good', 'Excellent'] = Field(description="Overall image quality")
    accessibility_description: str = Field(description="Description suitable for screen readers")


class DocumentImageAnalysis(BaseModel):
    """Analysis of document images with text extraction"""
    document_type: Literal['invoice', 'receipt', 'contract', 'form', 'letter', 'report', 'other'] = Field(
        description="Type of document"
    )
    extracted_text: str = Field(description="Text content extracted from the document")
    key_information: Dict[str, str] = Field(description="Key information fields extracted")
    language: str = Field(description="Primary language of the document")
    layout_analysis: str = Field(description="Analysis of document structure and layout")
    readability: Literal['Clear', 'Partially Clear', 'Difficult', 'Illegible'] = Field(
        description="Text readability assessment"
    )
    confidence: float = Field(ge=0, le=1, description="Overall extraction confidence")


class ChartDataExtraction(BaseModel):
    """Data extraction from charts and graphs"""
    chart_type: Literal['bar', 'line', 'pie', 'scatter', 'histogram', 'area', 'other'] = Field(
        description="Type of chart or graph"
    )
    title: Optional[str] = Field(None, description="Chart title if visible")
    x_axis_label: Optional[str] = Field(None, description="X-axis label")
    y_axis_label: Optional[str] = Field(None, description="Y-axis label")
    data_series: List[str] = Field(description="Names of data series or categories")
    key_insights: List[str] = Field(description="Key insights or trends visible in the chart")
    estimated_values: Dict[str, List[float]] = Field(description="Estimated data values extracted")
    data_summary: str = Field(description="Summary of the data represented")


class MedicalImageAnalysis(BaseModel):
    """Medical or health-related image analysis"""
    image_type: Literal['x-ray', 'mri', 'ct-scan', 'ultrasound', 'clinical-photo', 'microscopy', 'other'] = Field(
        description="Type of medical image"
    )
    anatomical_region: str = Field(description="Body region or anatomy shown")
    observations: List[str] = Field(description="Observable features or findings")
    technical_quality: Literal['Poor', 'Adequate', 'Good', 'Excellent'] = Field(
        description="Technical quality of the image"
    )
    artifacts: List[str] = Field(default=[], description="Any artifacts or image distortions noted")
    disclaimer: str = Field(
        default="This analysis is for informational purposes only and should not replace professional medical diagnosis.",
        description="Medical disclaimer"
    )


# ==============================================================================
# COMPOSITE AND COMPLEX MODELS
# ==============================================================================

class ComprehensiveAnalysis(BaseModel):
    """Multi-domain comprehensive analysis"""
    analysis_type: str = Field(description="Type of analysis performed")
    input_summary: str = Field(description="Summary of the input data/content")
    findings: List[str] = Field(description="Key findings from the analysis")
    recommendations: List[str] = Field(description="Actionable recommendations")
    risk_assessment: Literal['Low', 'Medium', 'High', 'Critical'] = Field(description="Risk level assessment")
    confidence_score: float = Field(ge=0, le=1, description="Overall confidence in the analysis")
    methodology: str = Field(description="Brief description of analysis methodology")
    limitations: List[str] = Field(default=[], description="Known limitations of the analysis")
    next_steps: List[str] = Field(description="Suggested next steps or follow-up actions")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata or context")


class ConversationSummary(BaseModel):
    """Conversation or dialogue summary"""
    participants: List[str] = Field(description="Names or identifiers of conversation participants")
    main_topics: List[str] = Field(description="Main topics discussed")
    key_points: List[str] = Field(description="Important points raised")
    decisions_made: List[str] = Field(default=[], description="Decisions or agreements reached")
    action_items: List[str] = Field(default=[], description="Tasks or actions assigned")
    sentiment_analysis: Dict[str, str] = Field(description="Sentiment for each participant")
    conversation_flow: str = Field(description="Overall flow and structure of the conversation")
    unresolved_issues: List[str] = Field(default=[], description="Issues left unresolved")


# ==============================================================================
# UTILITY MODELS
# ==============================================================================

class ValidationResult(BaseModel):
    """Result of data validation process"""
    is_valid: bool = Field(description="Whether the data passed validation")
    validation_score: float = Field(ge=0, le=1, description="Validation score")
    errors: List[str] = Field(default=[], description="Validation errors found")
    warnings: List[str] = Field(default=[], description="Validation warnings")
    corrected_data: Optional[Dict[str, Any]] = Field(None, description="Corrected version of the data")
    validation_criteria: List[str] = Field(description="Criteria used for validation")


class ComparisonResult(BaseModel):
    """Result of comparing two items or datasets"""
    item1_summary: str = Field(description="Summary of the first item")
    item2_summary: str = Field(description="Summary of the second item")
    similarities: List[str] = Field(description="Similarities between the items")
    differences: List[str] = Field(description="Differences between the items")
    comparison_score: float = Field(ge=0, le=1, description="Overall similarity score")
    winner: Optional[str] = Field(None, description="Which item is better, if applicable")
    recommendation: str = Field(description="Recommendation based on the comparison")


# ==============================================================================
# ENHANCED STRUCTURED MODELS
# ==============================================================================

class APIResponse(BaseModel):
    """Standardized API response structure"""
    success: bool = Field(description="Whether the operation was successful")
    data: Optional[Any] = Field(default=None, description="Response data")
    message: str = Field(description="Response message")
    error_code: Optional[str] = Field(default=None, description="Error code if applicable")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Request identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UserProfile(BaseModel):
    """User profile information"""
    user_id: str = Field(description="Unique user identifier")
    username: str = Field(description="Username")
    email: str = Field(description="Email address")
    full_name: Optional[str] = Field(default=None, description="Full name")
    avatar_url: Optional[str] = Field(default=None, description="Avatar image URL")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    created_at: datetime = Field(default_factory=datetime.now, description="Account creation date")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    is_active: bool = Field(default=True, description="Whether account is active")
    role: Literal["user", "admin", "moderator"] = Field(default="user", description="User role")
    
    @validator('email')
    def validate_email(cls, v):
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]{3,20}$', v):
            raise ValueError("Username must be 3-20 characters, alphanumeric with _ or -")
        return v


class FileMetadata(BaseModel):
    """File metadata structure"""
    file_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique file identifier")
    filename: str = Field(description="Original filename")
    file_path: str = Field(description="File path")
    file_size: int = Field(ge=0, description="File size in bytes")
    file_type: str = Field(description="MIME type or file extension")
    checksum: Optional[str] = Field(default=None, description="File checksum for integrity")
    created_at: datetime = Field(default_factory=datetime.now, description="File creation timestamp")
    modified_at: datetime = Field(default_factory=datetime.now, description="Last modification timestamp")
    uploaded_by: str = Field(description="User who uploaded the file")
    tags: List[str] = Field(default=[], description="File tags")
    description: Optional[str] = Field(default=None, description="File description")
    is_public: bool = Field(default=False, description="Whether file is publicly accessible")
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        # Remove dangerous characters
        v = re.sub(r'[<>:"/\\|?*]', '_', v)
        return v.strip()


class NotificationData(BaseModel):
    """Notification data structure"""
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Notification ID")
    user_id: str = Field(description="Target user ID")
    title: str = Field(description="Notification title")
    message: str = Field(description="Notification message")
    type: Literal["info", "warning", "error", "success", "system"] = Field(description="Notification type")
    priority: Literal["low", "medium", "high", "urgent"] = Field(default="medium", description="Priority level")
    category: str = Field(description="Notification category")
    is_read: bool = Field(default=False, description="Whether notification has been read")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    expires_at: Optional[datetime] = Field(default=None, description="Expiration timestamp")
    action_url: Optional[str] = Field(default=None, description="Action URL if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResult(BaseModel):
    """Search result structure"""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Result identifier")
    title: str = Field(description="Result title")
    content: str = Field(description="Result content")
    url: Optional[str] = Field(default=None, description="Result URL")
    source: str = Field(description="Result source")
    relevance_score: float = Field(ge=0, le=1, description="Relevance score")
    category: Optional[str] = Field(default=None, description="Result category")
    tags: List[str] = Field(default=[], description="Result tags")
    created_at: Optional[datetime] = Field(default=None, description="Content creation date")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SearchResponse(BaseModel):
    """Search response structure"""
    query: str = Field(description="Search query")
    total_results: int = Field(description="Total number of results")
    results: List[SearchResult] = Field(description="Search results")
    search_time_ms: float = Field(description="Search execution time in milliseconds")
    suggestions: List[str] = Field(default=[], description="Search suggestions")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    pagination: Dict[str, Any] = Field(default_factory=dict, description="Pagination info")


class PerformanceMetrics(BaseModel):
    """Performance metrics structure"""
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Metric identifier")
    component: str = Field(description="Component being measured")
    metric_name: str = Field(description="Metric name")
    value: float = Field(description="Metric value")
    unit: str = Field(description="Metric unit")
    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement timestamp")
    context: Dict[str, Any] = Field(default_factory=dict, description="Measurement context")
    tags: List[str] = Field(default=[], description="Metric tags")


class SystemHealth(BaseModel):
    """System health status"""
    status: Literal["healthy", "degraded", "unhealthy", "unknown"] = Field(description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    components: Dict[str, Dict[str, Any]] = Field(description="Individual component status")
    metrics: List[PerformanceMetrics] = Field(default=[], description="Health metrics")
    alerts: List[str] = Field(default=[], description="Active alerts")
    uptime_seconds: float = Field(description="System uptime in seconds")
    version: str = Field(description="System version")


class ConfigurationSchema(BaseModel):
    """Configuration schema structure"""
    config_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Configuration ID")
    name: str = Field(description="Configuration name")
    description: str = Field(description="Configuration description")
    version: str = Field(description="Configuration version")
    settings: Dict[str, Any] = Field(description="Configuration settings")
    environment: Literal["development", "staging", "production"] = Field(description="Target environment")
    is_active: bool = Field(default=True, description="Whether configuration is active")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    created_by: str = Field(description="Creator user ID")
    
    @validator('version')
    def validate_version(cls, v):
        # Basic semantic versioning validation
        version_pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(version_pattern, v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.0.0)")
        return v


class AuditLog(BaseModel):
    """Audit log entry"""
    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Audit log ID")
    user_id: str = Field(description="User who performed the action")
    action: str = Field(description="Action performed")
    resource: str = Field(description="Resource affected")
    resource_id: Optional[str] = Field(default=None, description="Resource identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Action timestamp")
    ip_address: Optional[str] = Field(default=None, description="IP address")
    user_agent: Optional[str] = Field(default=None, description="User agent")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")
    result: Literal["success", "failure", "partial"] = Field(description="Action result")
    severity: Literal["low", "medium", "high", "critical"] = Field(default="medium", description="Severity level")


# ==============================================================================
# MODEL REGISTRY
# ==============================================================================

# Registry of all available models for easy access
MODEL_REGISTRY = {
    # Basic Examples
    'PersonInfo': PersonInfo,
    'FriendInfo': FriendInfo,
    'FriendList': FriendList,
    
    # Image Analysis
    'DetectedObject': DetectedObject,
    'ImageDescription': ImageDescription,
    'VisualContentAnalysis': VisualContentAnalysis,
    'DocumentImageAnalysis': DocumentImageAnalysis,
    'ChartDataExtraction': ChartDataExtraction,
    'MedicalImageAnalysis': MedicalImageAnalysis,
    
    # Text Analysis
    'SentimentAnalysis': SentimentAnalysis,
    'CodeAnalysis': CodeAnalysis,
    'DocumentSummary': DocumentSummary,
    'QuestionAnswer': QuestionAnswer,
    'TranslationResult': TranslationResult,
    'DataExtractionResult': DataExtractionResult,
    'ProductReview': ProductReview,
    'MeetingNotes': MeetingNotes,
    
    # Complex Models
    'ComprehensiveAnalysis': ComprehensiveAnalysis,
    'ConversationSummary': ConversationSummary,
    'ValidationResult': ValidationResult,
    'ComparisonResult': ComparisonResult,
    
    # Enhanced Models
    'APIResponse': APIResponse,
    'UserProfile': UserProfile,
    'FileMetadata': FileMetadata,
    'NotificationData': NotificationData,
    'SearchResult': SearchResult,
    'SearchResponse': SearchResponse,
    'PerformanceMetrics': PerformanceMetrics,
    'SystemHealth': SystemHealth,
    'ConfigurationSchema': ConfigurationSchema,
    'AuditLog': AuditLog,
}


def get_model_by_name(model_name: str) -> BaseModel:
    """
    Get a Pydantic model by name from the registry
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        BaseModel: The requested Pydantic model class
        
    Raises:
        KeyError: If model name not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available_models = ', '.join(MODEL_REGISTRY.keys())
        raise KeyError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_name]


def list_available_models() -> List[str]:
    """
    Get list of all available model names
    
    Returns:
        List[str]: List of model names in the registry
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a specific model
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Dict[str, Any]: Model information including fields and descriptions
    """
    model_class = get_model_by_name(model_name)
    schema = model_class.model_json_schema()
    
    fields_info = {}
    if 'properties' in schema:
        for field_name, field_info in schema['properties'].items():
            fields_info[field_name] = {
                'type': field_info.get('type', 'unknown'),
                'description': field_info.get('description', ''),
                'required': field_name in schema.get('required', [])
            }
    
    return {
        'model_name': model_name,
        'class_name': model_class.__name__,
        'description': schema.get('description', model_class.__doc__ or ''),
        'fields': fields_info,
        'required_fields': schema.get('required', []),
        'total_fields': len(fields_info),
        'schema': schema
    }


# ==============================================================================
# MODEL CATEGORIES
# ==============================================================================

MODEL_CATEGORIES = {
    'Basic Examples': ['PersonInfo', 'FriendInfo', 'FriendList'],
    'Image Analysis': ['DetectedObject', 'ImageDescription', 'VisualContentAnalysis', 
                      'DocumentImageAnalysis', 'ChartDataExtraction', 'MedicalImageAnalysis'],
    'Text Analysis': ['SentimentAnalysis', 'CodeAnalysis', 'DocumentSummary', 
                     'QuestionAnswer', 'TranslationResult', 'DataExtractionResult'],
    'Business & Productivity': ['ProductReview', 'MeetingNotes'],
    'Complex Analysis': ['ComprehensiveAnalysis', 'ConversationSummary'],
    'Utility Models': ['ValidationResult', 'ComparisonResult'],
    'API & System': ['APIResponse', 'SystemHealth', 'ConfigurationSchema', 'AuditLog'],
    'User Management': ['UserProfile', 'NotificationData'],
    'File & Data': ['FileMetadata', 'SearchResult', 'SearchResponse'],
    'Performance': ['PerformanceMetrics']
}


def get_models_by_category(category: str) -> List[str]:
    """
    Get models in a specific category
    
    Args:
        category (str): Category name
        
    Returns:
        List[str]: List of model names in the category
    """
    return MODEL_CATEGORIES.get(category, [])


def get_all_categories() -> List[str]:
    """
    Get all available model categories
    
    Returns:
        List[str]: List of category names
    """
    return list(MODEL_CATEGORIES.keys())


# ==============================================================================
# ENHANCED UTILITY FUNCTIONS
# ==============================================================================

def create_api_response(success: bool, message: str, data: Any = None, 
                       error_code: str = None, metadata: Dict[str, Any] = None) -> APIResponse:
    """Create a standardized API response"""
    return APIResponse(
        success=success,
        message=message,
        data=data,
        error_code=error_code,
        metadata=metadata or {}
    )


def create_success_response(message: str, data: Any = None, metadata: Dict[str, Any] = None) -> APIResponse:
    """Create a success API response"""
    return create_api_response(True, message, data, None, metadata)


def create_error_response(message: str, error_code: str = None, metadata: Dict[str, Any] = None) -> APIResponse:
    """Create an error API response"""
    return create_api_response(False, message, None, error_code, metadata)


def validate_structured_output(data: Dict[str, Any], model_name: str) -> bool:
    """Validate data against a structured model"""
    try:
        model_class = get_model_by_name(model_name)
        model_class(**data)
        return True
    except Exception:
        return False


def convert_to_structured_output(data: Dict[str, Any], model_name: str) -> Optional[BaseModel]:
    """Convert data to structured output using specified model"""
    try:
        model_class = get_model_by_name(model_name)
        return model_class(**data)
    except Exception:
        return None


def extract_structured_fields(data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Extract fields that match a structured model schema"""
    try:
        model_class = get_model_by_name(model_name)
        schema = model_class.model_json_schema()
        required_fields = schema.get('required', [])
        properties = schema.get('properties', {})
        
        extracted = {}
        for field_name, field_info in properties.items():
            if field_name in data:
                extracted[field_name] = data[field_name]
        
        return extracted
    except Exception:
        return {}


def get_model_field_info(model_name: str) -> Dict[str, Any]:
    """Get detailed field information for a model"""
    try:
        model_class = get_model_by_name(model_name)
        schema = model_class.model_json_schema()
        
        field_info = {}
        for field_name, field_schema in schema.get('properties', {}).items():
            field_info[field_name] = {
                'type': field_schema.get('type', 'unknown'),
                'description': field_schema.get('description', ''),
                'required': field_name in schema.get('required', []),
                'default': field_schema.get('default'),
                'constraints': {
                    'min_length': field_schema.get('minLength'),
                    'max_length': field_schema.get('maxLength'),
                    'minimum': field_schema.get('minimum'),
                    'maximum': field_schema.get('maximum'),
                    'pattern': field_schema.get('pattern')
                }
            }
        
        return field_info
    except Exception:
        return {}


def generate_model_example(model_name: str) -> Dict[str, Any]:
    """Generate an example data structure for a model"""
    try:
        model_class = get_model_by_name(model_name)
        schema = model_class.model_json_schema()
        
        example = {}
        for field_name, field_schema in schema.get('properties', {}).items():
            field_type = field_schema.get('type', 'string')
            
            if field_type == 'string':
                example[field_name] = field_schema.get('description', f'Example {field_name}')
            elif field_type == 'integer':
                example[field_name] = 0
            elif field_type == 'number':
                example[field_name] = 0.0
            elif field_type == 'boolean':
                example[field_name] = True
            elif field_type == 'array':
                example[field_name] = []
            elif field_type == 'object':
                example[field_name] = {}
            else:
                example[field_name] = None
        
        return example
    except Exception:
        return {}
