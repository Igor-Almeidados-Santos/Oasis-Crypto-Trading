#!/usr/bin/env python3
"""
Oasis Crypto Trade - Secrets Generator
======================================

Secure generation of application secrets and API keys for development and production.

Features:
- Cryptographically secure random generation
- Environment-specific configurations
- Secure key derivation functions
- JWT key pair generation
- Database and cache passwords
- API key generation

Author: Oasis Trading Systems
License: Proprietary
"""

import os
import secrets
import string
import sys
from pathlib import Path
from typing import Dict, Any
import base64
import hashlib
from datetime import datetime

# Cryptographic imports
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    print("Warning: cryptography library not installed. Some features will be limited.")


# =============================================================================
# CONFIGURATION
# =============================================================================

class SecretConfig:
    """Configuration for secret generation."""
    
    # Secret lengths
    SECRET_KEY_LENGTH = 64
    API_KEY_LENGTH = 32
    PASSWORD_LENGTH = 24
    JWT_SECRET_LENGTH = 64
    
    # Character sets
    ALPHANUM = string.ascii_letters + string.digits
    SAFE_CHARS = string.ascii_letters + string.digits + "!@#$%^&*"
    HEX_CHARS = string.hexdigits.lower()
    
    # Key derivation settings
    PBKDF2_ITERATIONS = 100000
    SALT_LENGTH = 32


# =============================================================================
# SECRET GENERATORS
# =============================================================================

class OasisSecretGenerator:
    """Secure secret generation utility."""
    
    def __init__(self, config: SecretConfig = None):
        self.config = config or SecretConfig()
        
        # Ensure cryptographically secure random source
        self.secure_random = secrets.SystemRandom()
    
    def generate_secret_key(self, length: int = None) -> str:
        """Generate cryptographically secure secret key."""
        length = length or self.config.SECRET_KEY_LENGTH
        return secrets.token_urlsafe(length)
    
    def generate_password(
        self, 
        length: int = None, 
        include_symbols: bool = True
    ) -> str:
        """Generate secure password."""
        length = length or self.config.PASSWORD_LENGTH
        
        if include_symbols:
            chars = self.config.SAFE_CHARS
        else:
            chars = self.config.ALPHANUM
        
        # Ensure at least one character from each category
        password = []
        
        # At least one lowercase
        password.append(self.secure_random.choice(string.ascii_lowercase))
        
        # At least one uppercase
        password.append(self.secure_random.choice(string.ascii_uppercase))
        
        # At least one digit
        password.append(self.secure_random.choice(string.digits))
        
        if include_symbols:
            # At least one symbol
            symbols = "!@#$%^&*"
            password.append(self.secure_random.choice(symbols))
        
        # Fill remaining length with random characters
        for _ in range(length - len(password)):
            password.append(self.secure_random.choice(chars))
        
        # Shuffle the password
        self.secure_random.shuffle(password)
        
        return ''.join(password)
    
    def generate_api_key(self, prefix: str = "oasis") -> str:
        """Generate API key with prefix."""
        key_part = secrets.token_urlsafe(self.config.API_KEY_LENGTH)
        return f"{prefix}_{key_part}"
    
    def generate_hex_key(self, length: int = 32) -> str:
        """Generate hexadecimal key."""
        return secrets.token_hex(length)
    
    def generate_uuid(self) -> str:
        """Generate UUID4."""
        import uuid
        return str(uuid.uuid4())
    
    def generate_jwt_secret(self, length: int = None) -> str:
        """Generate JWT signing secret."""
        length = length or self.config.JWT_SECRET_LENGTH
        return secrets.token_urlsafe(length)
    
    def generate_salt(self, length: int = None) -> str:
        """Generate cryptographic salt."""
        length = length or self.config.SALT_LENGTH
        return secrets.token_hex(length)
    
    def derive_key(self, password: str, salt: bytes = None) -> str:
        """Derive key from password using PBKDF2."""
        if not HAS_CRYPTOGRAPHY:
            # Fallback to simple hash
            return hashlib.sha256(password.encode()).hexdigest()
        
        if salt is None:
            salt = os.urandom(self.config.SALT_LENGTH)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.PBKDF2_ITERATIONS,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode()
    
    def generate_rsa_key_pair(self, key_size: int = 2048) -> Dict[str, str]:
        """Generate RSA key pair for JWT signing."""
        if not HAS_CRYPTOGRAPHY:
            raise ImportError("cryptography library required for RSA key generation")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
        
        # Serialize public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
        
        return {
            'private_key': private_pem,
            'public_key': public_pem
        }


# =============================================================================
# ENVIRONMENT CONFIGURATION GENERATORS
# =============================================================================

class EnvironmentConfigGenerator:
    """Generate environment-specific configurations."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.generator = OasisSecretGenerator()
    
    def generate_database_config(self) -> Dict[str, str]:
        """Generate database configuration."""
        return {
            'POSTGRES_PASSWORD': self.generator.generate_password(32, include_symbols=False),
            'POSTGRES_DB': f'oasis_trading_db_{self.environment}',
            'POSTGRES_USER': 'oasis_admin',
        }
    
    def generate_cache_config(self) -> Dict[str, str]:
        """Generate cache configuration."""
        return {
            'REDIS_PASSWORD': self.generator.generate_password(24, include_symbols=False),
        }
    
    def generate_security_config(self) -> Dict[str, str]:
        """Generate security configuration."""
        config = {
            'SECRET_KEY': self.generator.generate_secret_key(),
            'JWT_SECRET_KEY': self.generator.generate_jwt_secret(),
            'ENCRYPTION_KEY': self.generator.generate_hex_key(32),
            'SESSION_SECRET': self.generator.generate_secret_key(32),
        }
        
        # Generate RSA key pair for JWT if available
        if HAS_CRYPTOGRAPHY:
            try:
                rsa_keys = self.generator.generate_rsa_key_pair()
                config.update({
                    'JWT_PRIVATE_KEY': rsa_keys['private_key'],
                    'JWT_PUBLIC_KEY': rsa_keys['public_key'],
                })
            except Exception as e:
                print(f"Warning: Could not generate RSA keys: {e}")
        
        return config
    
    def generate_api_keys(self) -> Dict[str, str]:
        """Generate API keys for different services."""
        return {
            'OASIS_API_KEY': self.generator.generate_api_key('oasis'),
            'INTERNAL_API_KEY': self.generator.generate_api_key('internal'),
            'WEBHOOK_SECRET': self.generator.generate_secret_key(32),
            'MONITORING_TOKEN': self.generator.generate_api_key('monitor'),
        }
    
    def generate_kafka_config(self) -> Dict[str, str]:
        """Generate Kafka configuration."""
        return {
            'KAFKA_SASL_USERNAME': 'oasis_kafka',
            'KAFKA_SASL_PASSWORD': self.generator.generate_password(20, include_symbols=False),
        }
    
    def generate_all_secrets(self) -> Dict[str, str]:
        """Generate all secrets for the environment."""
        all_secrets = {}
        
        # Database secrets
        all_secrets.update(self.generate_database_config())
        
        # Cache secrets
        all_secrets.update(self.generate_cache_config())
        
        # Security secrets
        all_secrets.update(self.generate_security_config())
        
        # API keys
        all_secrets.update(self.generate_api_keys())
        
        # Kafka secrets
        all_secrets.update(self.generate_kafka_config())
        
        # Environment-specific settings
        all_secrets.update({
            'ENVIRONMENT': self.environment,
            'DEBUG': 'true' if self.environment == 'development' else 'false',
            'TESTING': 'true' if self.environment == 'testing' else 'false',
        })
        
        return all_secrets


# =============================================================================
# FILE WRITERS
# =============================================================================

def write_env_file(secrets: Dict[str, str], file_path: Path):
    """Write secrets to .env file."""
    with open(file_path, 'w') as f:
        f.write("# =============================================================================\n")
        f.write("# Oasis Crypto Trade - Environment Configuration\n")
        f.write("# =============================================================================\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write("# WARNING: Keep this file secure and never commit to version control!\n")
        f.write("# =============================================================================\n\n")
        
        # Group related settings
        sections = {
            'APPLICATION': ['ENVIRONMENT', 'DEBUG', 'TESTING'],
            'SECURITY': ['SECRET_KEY', 'JWT_SECRET_KEY', 'JWT_PRIVATE_KEY', 'JWT_PUBLIC_KEY', 
                        'ENCRYPTION_KEY', 'SESSION_SECRET'],
            'DATABASE': ['POSTGRES_PASSWORD', 'POSTGRES_DB', 'POSTGRES_USER'],
            'CACHE': ['REDIS_PASSWORD'],
            'MESSAGING': ['KAFKA_SASL_USERNAME', 'KAFKA_SASL_PASSWORD'],
            'API_KEYS': ['OASIS_API_KEY', 'INTERNAL_API_KEY', 'WEBHOOK_SECRET', 'MONITORING_TOKEN'],
        }
        
        for section, keys in sections.items():
            f.write(f"# {section}\n")
            for key in keys:
                if key in secrets:
                    value = secrets[key]
                    # Handle multiline values (like RSA keys)
                    if '\n' in value:
                        f.write(f'{key}="""\n{value}"""\n')
                    else:
                        f.write(f'{key}={value}\n')
            f.write('\n')


def write_secrets_summary(secrets: Dict[str, str], file_path: Path):
    """Write secrets summary (without sensitive values)."""
    with open(file_path, 'w') as f:
        f.write("Oasis Crypto Trade - Secrets Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Total secrets: {len(secrets)}\n\n")
        
        for key in sorted(secrets.keys()):
            if 'KEY' in key or 'PASSWORD' in key or 'SECRET' in key:
                f.write(f"{key}: [REDACTED]\n")
            else:
                f.write(f"{key}: {secrets[key]}\n")


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def generate_secrets_for_environment(environment: str = 'development'):
    """Generate secrets for specific environment."""
    print(f"🔐 Generating secrets for {environment} environment...")
    
    # Create generator
    config_gen = EnvironmentConfigGenerator(environment)
    
    # Generate all secrets
    secrets = config_gen.generate_all_secrets()
    
    # Determine file paths
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / f'.env.{environment}'
    summary_file = project_root / f'secrets_summary_{environment}.txt'
    
    # Write files
    write_env_file(secrets, env_file)
    write_secrets_summary(secrets, summary_file)
    
    print(f"✅ Secrets generated:")
    print(f"   📄 Environment file: {env_file}")
    print(f"   📋 Summary file: {summary_file}")
    print(f"   🔑 Total secrets: {len(secrets)}")
    
    # Set secure file permissions
    try:
        os.chmod(env_file, 0o600)  # Owner read/write only
        print(f"   🔒 Secure permissions set on {env_file}")
    except Exception as e:
        print(f"   ⚠️  Could not set secure permissions: {e}")
    
    return secrets


def validate_existing_secrets(env_file: Path) -> bool:
    """Validate existing secrets file."""
    if not env_file.exists():
        return False
    
    print(f"🔍 Validating existing secrets in {env_file}...")
    
    required_keys = [
        'SECRET_KEY', 'JWT_SECRET_KEY', 'POSTGRES_PASSWORD', 
        'REDIS_PASSWORD', 'OASIS_API_KEY'
    ]
    
    existing_secrets = {}
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_secrets[key] = value
        
        missing_keys = [key for key in required_keys if key not in existing_secrets]
        
        if missing_keys:
            print(f"❌ Missing required keys: {missing_keys}")
            return False
        
        weak_secrets = []
        for key, value in existing_secrets.items():
            if 'SECRET' in key or 'PASSWORD' in key or 'KEY' in key:
                if len(value) < 16:
                    weak_secrets.append(key)
        
        if weak_secrets:
            print(f"⚠️  Weak secrets detected: {weak_secrets}")
            return False
        
        print("✅ All secrets validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating secrets: {e}")
        return False


def main():
    """Main function."""
    print("🏛️ Oasis Crypto Trade - Secrets Generator")
    print("=" * 50)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        environment = sys.argv[1]
    else:
        environment = 'development'
    
    valid_environments = ['development', 'testing', 'staging', 'production']
    if environment not in valid_environments:
        print(f"❌ Invalid environment. Choose from: {valid_environments}")
        sys.exit(1)
    
    # Check for existing secrets
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / f'.env.{environment}'
    
    if env_file.exists():
        print(f"📁 Existing secrets file found: {env_file}")
        
        if validate_existing_secrets(env_file):
            response = input("Valid secrets exist. Regenerate? [y/N]: ")
            if response.lower() != 'y':
                print("✋ Keeping existing secrets")
                sys.exit(0)
        else:
            print("🔄 Invalid secrets detected. Regenerating...")
    
    # Generate secrets
    try:
        secrets = generate_secrets_for_environment(environment)
        
        # Special handling for production
        if environment == 'production':
            print("\n🚨 PRODUCTION ENVIRONMENT DETECTED 🚨")
            print("   • Store secrets securely (use a secret manager)")
            print("   • Never commit .env.production to version control")
            print("   • Regularly rotate secrets")
            print("   • Monitor for unauthorized access")
        
        print(f"\n✅ Secret generation completed for {environment}")
        
    except Exception as e:
        print(f"❌ Error generating secrets: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()