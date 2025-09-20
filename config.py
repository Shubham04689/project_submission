"""
Configuration file for the LLM Agent
Real API integration with Groq, Mistral, and Ollama
"""

import os
from typing import Optional

# API Configuration
# Groq API key saved directly in application
DEFAULT_GROQ_API_KEY = "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"

GROQ_API_KEY = os.getenv('GROQ_API_KEY', DEFAULT_GROQ_API_KEY)
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model Configuration
GROQ_MODELS = {
    'fast': 'llama-3.1-8b-instant',
    'balanced': 'llama-3.1-70b-versatile', 
    'creative': 'llama-3.1-70b-versatile'
}

MISTRAL_MODELS = {
    'fast': 'mistral-tiny',
    'balanced': 'mistral-small',
    'creative': 'mistral-medium'
}

OLLAMA_MODELS = {
    'fast': 'gemma3:4b',
    'balanced': 'llama3.2:8b',
    'creative': 'codellama:7b'
}

# Agent Configuration
DEFAULT_MEMORY_SIZE = 3
DEFAULT_TEMPERATURE = 0.7
MAX_TOKENS = 150
DEFAULT_PROVIDER = 'groq'  # 'groq', 'mistral', 'ollama'
DEFAULT_MODEL = 'llama-3.1-8b-instant'  # Default Llama 3 model

# Intent Detection Configuration
FACTUAL_CONFIDENCE_THRESHOLD = 0.6
CREATIVE_CONFIDENCE_THRESHOLD = 0.6

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for specified provider"""
    keys = {
        'groq': GROQ_API_KEY,
        'mistral': MISTRAL_API_KEY,
        'openai': OPENAI_API_KEY
    }
    return keys.get(provider.lower())

def get_model_for_task(provider: str, task_type: str = 'balanced') -> str:
    """Get appropriate model for provider and task type"""
    models = {
        'groq': GROQ_MODELS,
        'mistral': MISTRAL_MODELS,
        'ollama': OLLAMA_MODELS
    }
    
    provider_models = models.get(provider.lower(), GROQ_MODELS)
    return provider_models.get(task_type, list(provider_models.values())[0])