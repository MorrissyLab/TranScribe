from abc import ABC, abstractmethod
from typing import Any

class BaseLLMFactory(ABC):
    """Abstract base class for LLM instantiation factories."""
    
    @abstractmethod
    def get_llm(self, model_name: str, temperature: float) -> Any:
        """Instantiates and returns the LLM chat model."""
        pass

class GeminiLLMFactory(BaseLLMFactory):
    """Factory for Google Gemini models via LangChain."""
    
    def get_llm(self, model_name: str, temperature: float) -> Any:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        except ImportError:
            raise ImportError("Please install langchain-google-genai to use Gemini models.")

class OpenAILLMFactory(BaseLLMFactory):
    """Factory for OpenAI models via LangChain."""
    
    def get_llm(self, model_name: str, temperature: float) -> Any:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=temperature)
        except ImportError:
            raise ImportError("Please install langchain-openai to use OpenAI models.")

class LLMFactory:
    """Registry and access point for LLM providers."""
    
    _factories = {
        "gemini": GeminiLLMFactory(),
        "openai": OpenAILLMFactory()
    }
    _instance_cache = {}
    
    @classmethod
    def register_provider(cls, provider_name: str, factory: BaseLLMFactory):
        """Allows registering new custom providers at runtime."""
        cls._factories[provider_name.lower()] = factory
        
    @classmethod
    def get_provider(cls, provider_name: str) -> BaseLLMFactory:
        """Retrieves the factory instance for the requested provider."""
        factory = cls._factories.get(provider_name.lower())
        if not factory:
            raise ValueError(f"Unsupported LLM provider: {provider_name}. Available: {list(cls._factories.keys())}")
        return factory

    @classmethod
    def infer_provider(cls, model_name: str) -> str:
        """Infers the appropriate provider based on the model name."""
        name = model_name.lower()
        if name.startswith(("gpt-", "o1-")):
            return "openai"
        if name.startswith(("gemini-", "gemma-")):
            return "gemini"
        # Default or fallback cases could be added here
        return "openai" # Default to openai if unsure, or raise error

    @classmethod
    def get_llm(cls, provider_name: str, model_name: str, temperature: float = 0.1) -> Any:
        """Retrieves a cached LLM instance or creates a new one."""
        cache_key = (provider_name.lower(), model_name, temperature)
        if cache_key not in cls._instance_cache:
            factory = cls.get_provider(provider_name)
            cls._instance_cache[cache_key] = factory.get_llm(model_name, temperature)
        return cls._instance_cache[cache_key]
