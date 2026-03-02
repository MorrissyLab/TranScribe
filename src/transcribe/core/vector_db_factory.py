from abc import ABC, abstractmethod
from typing import Any, List, Dict
from transcribe.core.embeddings_factory import EmbeddingsFactory

class BaseVectorDBFactory(ABC):
    """Abstract base class for Vector DB instantiation."""
    
    @abstractmethod
    def get_vector_db(self, index_name: str, embeddings_provider: str, embeddings_model: str) -> Any:
        """Instantiates and returns the Vector DB object."""
        pass

class PineconeVectorDBFactory(BaseVectorDBFactory):
    """Factory for Pinecone Vector DB via LangChain."""
    
    def get_vector_db(self, index_name: str, embeddings_provider: str = "gemini", embeddings_model: str = "models/text-embedding-004") -> Any:
        try:
            from langchain_pinecone import PineconeVectorStore
            import pinecone
        except ImportError:
            raise ImportError("Please install pinecone-client and langchain-pinecone to use Pinecone.")
            
        embeddings = EmbeddingsFactory.get_provider(embeddings_provider).get_embeddings(embeddings_model)
        
        # We assume PINECONE_API_KEY environment variable is set as required by the library
        return PineconeVectorStore(index_name=index_name, embedding=embeddings)

class VectorDBFactory:
    """Registry and access point for Vector DB providers."""
    
    _factories = {
        "pinecone": PineconeVectorDBFactory()
    }
    
    @classmethod
    def register_provider(cls, provider_name: str, factory: BaseVectorDBFactory):
        """Allows registering new custom providers at runtime."""
        cls._factories[provider_name.lower()] = factory
        
    @classmethod
    def get_provider(cls, provider_name: str) -> BaseVectorDBFactory:
        """Retrieves the factory instance for the requested provider."""
        factory = cls._factories.get(provider_name.lower())
        if not factory:
            raise ValueError(f"Unsupported Vector DB provider: {provider_name}. Available: {list(cls._factories.keys())}")
        return factory
