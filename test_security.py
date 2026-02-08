"""
Test file to trigger critical ethics issues in the analyzer.
This file intentionally contains security vulnerabilities for testing.
"""

import requests
import os

# CRITICAL: Hardcoded API credentials (will be detected!)
API_KEY = "sk-1234567890abcdefghijklmnop"
SECRET_TOKEN = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
DATABASE_PASSWORD = "MySecretPass123!"

# More critical issues
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
STRIPE_SECRET_KEY = "sk_live_51234567890abcdefghijk"

def connect_to_database():
    """Connect to database with hardcoded password."""
    # CRITICAL: Password in code
    db_password = "admin123"
    connection_string = f"mongodb://admin:{db_password}@localhost:27017"
    return connection_string

def api_request():
    """Make API request with hardcoded token."""
    # CRITICAL: API key exposed
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-API-Token": "prod_token_987654321"
    }
    response = requests.get("https://api.example.com/data", headers=headers)
    return response.json()

# CRITICAL: Multiple credential patterns
config = {
    "api_key": "AIzaSyD1234567890abcdefghijklmnop",
    "secret": "my_secret_key_12345",
    "password": "SuperSecret2024!",
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
}

class DatabaseConfig:
    """Database configuration with hardcoded credentials."""
    
    # CRITICAL: Credentials in class
    USERNAME = "admin"
    PWD = "database_password_123"
    API_SECRET = "secret_api_key_prod_2024"
    
    def get_connection(self):
        """Get database connection."""
        # CRITICAL: Password hardcoded
        return f"Server=localhost;Database=mydb;User={self.USERNAME};Password={self.PWD}"

# CRITICAL: More patterns to detect
GITHUB_TOKEN = "ghp_16CharactersLongTokenExample123"
OPENAI_KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"

def main():
    """Main function with security issues."""
    # Testing credentials exposure
    print(f"Connecting with API key: {API_KEY}")
    print(f"Database password: {DATABASE_PASSWORD}")
    
    # This should trigger multiple critical issues
    pass

if __name__ == "__main__":
    main()
