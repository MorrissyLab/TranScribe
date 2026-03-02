from typing import List, Dict, Any
from transcribe.core.vector_db_factory import VectorDBFactory
from transcribe.config import logger

def retrieve_rag_context(
    query: str, 
    metadata_filters: Dict[str, Any],
    index_name: str,
    top_k: int = 5,
    vector_db_provider: str = "pinecone",
    embeddings_provider: str = "gemini",
    embeddings_model: str = "models/text-embedding-004"
) -> str:
    """
    Queries the Vector DB combining similarity search on `query` with `metadata_filters`.
    
    Expected metadata_filters format depends on the underlying provider, but LangChain
    mostly standardizes it (e.g., {"organism": "Human", "tissue": "PBMC"}).
    If `genes` are needed, you might need a `$in` MongoDB-style filter for Pinecone.
    """
    logger.info(f"Retrieving RAG context from {vector_db_provider} (index: {index_name})")
    logger.debug(f"Query: {query}")
    logger.debug(f"Filters: {metadata_filters}")
    
    try:
        vdb_factory = VectorDBFactory.get_provider(vector_db_provider)
        vector_store = vdb_factory.get_vector_db(index_name, embeddings_provider, embeddings_model)
        
        # Perform similarity search with metadata filtering
        docs = vector_store.similarity_search(query, k=top_k, filter=metadata_filters)
        
        if not docs:
            logger.info("No matching RAG documents found.")
            return "No relevant background knowledge retrieved."
            
        # Format the retrieved documents into a single context string to inject into the prompt
        context_parts = []
        for i, doc in enumerate(docs):
            source_genes = doc.metadata.get("genes", [])
            context_parts.append(f"[Source {i+1} - Genes: {source_genes}]: {doc.page_content}")
            
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Failed to retrieve RAG context: {e}")
        return "Error occurred while retrieving RAG context."
