"""
Security utilities for DurgasAI
Handles encryption, decryption, and secure storage of sensitive data
"""

import base64
import hashlib
import secrets
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional, Dict, Any
from .logging_utils import log_info, log_error, log_warning

class SecurityManager:
    """Manages encryption and security for sensitive data"""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password
        self._fernet = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with master password or generate new key"""
        try:
            if self.master_password:
                # Derive key from master password
                key = self._derive_key_from_password(self.master_password)
            else:
                # Generate a new random key
                key = Fernet.generate_key()
            
            self._fernet = Fernet(key)
            log_info("Encryption initialized successfully", "security")
        except Exception as e:
            log_error("Failed to initialize encryption", "security", e)
            self._fernet = None
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        password_bytes = password.encode('utf-8')
        salt = b'durgasai_salt_2024'  # In production, use a random salt per user
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt an API key"""
        if not self._fernet:
            log_warning("Encryption not initialized, storing in plain text", "security")
            return api_key
        
        try:
            encrypted_key = self._fernet.encrypt(api_key.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted_key).decode('utf-8')
        except Exception as e:
            log_error("Failed to encrypt API key", "security", e)
            return api_key
    
    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt an API key"""
        if not self._fernet:
            return encrypted_key
        
        try:
            # Check if the key looks like it might be encrypted
            if not self.is_encrypted(encrypted_key):
                log_warning("API key does not appear to be encrypted, returning as-is", "security")
                return encrypted_key
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_key.encode('utf-8'))
            decrypted_key = self._fernet.decrypt(encrypted_bytes)
            return decrypted_key.decode('utf-8')
        except Exception as e:
            log_error("Failed to decrypt API key - key may be corrupted or encrypted with different key", "security", e)
            # Return empty string instead of the corrupted encrypted key
            return ""
    
    def mask_api_key(self, api_key: str, visible_chars: int = 4) -> str:
        """Mask an API key for display purposes"""
        if not api_key or len(api_key) <= visible_chars:
            return "*" * len(api_key) if api_key else ""
        
        return api_key[:visible_chars] + "*" * (len(api_key) - visible_chars)
    
    def is_encrypted(self, text: str) -> bool:
        """Check if text appears to be encrypted"""
        if not text or not text.strip():
            return False
        
        try:
            # Check if it's valid base64 and has reasonable length for encrypted data
            decoded = base64.urlsafe_b64decode(text.encode('utf-8'))
            # Encrypted data should be at least 32 bytes (Fernet minimum)
            return len(decoded) >= 32
        except:
            return False
    
    def hash_api_key(self, api_key: str) -> str:
        """Create a hash of API key for verification without storing the key"""
        return hashlib.sha256(api_key.encode('utf-8')).hexdigest()
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token"""
        return secrets.token_urlsafe(length)

class APISecurityManager:
    """Specialized security manager for API keys"""
    
    def __init__(self):
        self.security_manager = SecurityManager()
    
    def secure_api_keys(self, api_keys: Dict[str, str], encrypt: bool = True) -> Dict[str, str]:
        """Secure a dictionary of API keys"""
        secured_keys = {}
        
        for provider, key in api_keys.items():
            if not key or not key.strip():
                secured_keys[provider] = ""
                continue
            
            if encrypt and self.security_manager.is_encrypted(key):
                # Already encrypted
                secured_keys[provider] = key
            elif encrypt:
                # Encrypt the key
                secured_keys[provider] = self.security_manager.encrypt_api_key(key.strip())
            else:
                # Just store as-is
                secured_keys[provider] = key.strip()
        
        return secured_keys
    
    def unsecure_api_keys(self, api_keys: Dict[str, str]) -> Dict[str, str]:
        """Unsecure (decrypt) a dictionary of API keys"""
        unsecured_keys = {}
        
        for provider, key in api_keys.items():
            if not key or not key.strip():
                unsecured_keys[provider] = ""
                continue
            
            if self.security_manager.is_encrypted(key):
                # Decrypt the key
                decrypted_key = self.security_manager.decrypt_api_key(key)
                unsecured_keys[provider] = decrypted_key if decrypted_key else ""
            else:
                # Already unencrypted
                unsecured_keys[provider] = key
        
        return unsecured_keys
    
    def mask_api_keys_for_display(self, api_keys: Dict[str, str]) -> Dict[str, str]:
        """Mask API keys for safe display"""
        masked_keys = {}
        
        for provider, key in api_keys.items():
            if not key or not key.strip():
                masked_keys[provider] = ""
            else:
                masked_keys[provider] = self.security_manager.mask_api_key(key)
        
        return masked_keys
    
    def validate_api_key_security(self, api_key: str) -> Dict[str, Any]:
        """Validate API key security characteristics"""
        if not api_key or not api_key.strip():
            return {
                "is_valid": False,
                "length": 0,
                "has_uppercase": False,
                "has_lowercase": False,
                "has_digits": False,
                "has_special_chars": False,
                "security_score": 0
            }
        
        key = api_key.strip()
        length = len(key)
        has_uppercase = any(c.isupper() for c in key)
        has_lowercase = any(c.islower() for c in key)
        has_digits = any(c.isdigit() for c in key)
        has_special_chars = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in key)
        
        # Calculate security score (0-100)
        security_score = 0
        if length >= 20:
            security_score += 30
        elif length >= 10:
            security_score += 20
        else:
            security_score += 10
        
        if has_uppercase:
            security_score += 20
        if has_lowercase:
            security_score += 20
        if has_digits:
            security_score += 15
        if has_special_chars:
            security_score += 15
        
        return {
            "is_valid": length >= 10 and security_score >= 50,
            "length": length,
            "has_uppercase": has_uppercase,
            "has_lowercase": has_lowercase,
            "has_digits": has_digits,
            "has_special_chars": has_special_chars,
            "security_score": min(security_score, 100)
        }
