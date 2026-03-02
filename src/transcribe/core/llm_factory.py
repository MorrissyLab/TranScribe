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
