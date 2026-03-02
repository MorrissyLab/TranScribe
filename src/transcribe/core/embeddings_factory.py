from abc import ABC, abstractmethod
from typing import Any

class BaseEmbeddingsFactory(ABC):
    """Abstract base class for Embeddings instantiation."""
    
    @abstractmethod
    def get_embeddings(self, model_name: str) -> Any:
        """Instantiates and returns the embeddings object."""
        pass

class GeminiEmbeddingsFactory(BaseEmbeddingsFactory):
    """Factory for Google Gemini embeddings via LangChain."""
    
    def get_embeddings(self, model_name: str = "models/text-embedding-004") -> Any:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(model=model_name)
        except ImportError:
            raise ImportError("Please install langchain-google-genai to use Gemini embeddings.")

class OpenAIEmbeddingsFactory(BaseEmbeddingsFactory):
    """Factory for OpenAI embeddings via LangChain."""
    
    def get_embeddings(self, model_name: str = "text-embedding-3-small") -> Any:
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name)
        except ImportError:
            raise ImportError("Please install langchain-openai to use OpenAI embeddings.")

class EmbeddingsFactory:
    """Registry and access point for Embeddings providers."""
    
    _factories = {
        "gemini": GeminiEmbeddingsFactory(),
        "openai": OpenAIEmbeddingsFactory()
    }
    
    @classmethod
    def register_provider(cls, provider_name: str, factory: BaseEmbeddingsFactory):
        """Allows registering new custom providers at runtime."""
        cls._factories[provider_name.lower()] = factory
        
    @classmethod
    def get_provider(cls, provider_name: str) -> BaseEmbeddingsFactory:
        """Retrieves the factory instance for the requested provider."""
        factory = cls._factories.get(provider_name.lower())
        if not factory:
            raise ValueError(f"Unsupported Embeddings provider: {provider_name}. Available: {list(cls._factories.keys())}")
        return factory
