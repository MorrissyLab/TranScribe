import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document
from transcribe.core.vector_db_factory import VectorDBFactory
from transcribe.config import logger

def ingest_gene_data(
    jsonl_path: str, 
    index_name: str, 
    vector_db_provider: str = "pinecone",
    embeddings_provider: str = "gemini", 
    embeddings_model: str = "models/text-embedding-004"
):
    """
    Ingests structural gene data (JSONL format) into the configured Vector DB.
    
    Expected JSONL structure:
    {
       "text": "Gene XYZ is a transcription factor...",
       "metadata": {
           "genes": ["XYZ", "ABC"],
           "organism": "Human",
           "tissue": "PBMC"
       }
    }
    """
    logger.info(f"Starting ingestion from {jsonl_path} into {vector_db_provider} (index: {index_name})")
    
    vdb_factory = VectorDBFactory.get_provider(vector_db_provider)
    # The factory returns a LangChain VectorStore (e.g. PineconeVectorStore)
    vector_store = vdb_factory.get_vector_db(index_name, embeddings_provider, embeddings_model)
    
    documents = []
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get("text", "")
                metadata = data.get("metadata", {})
                
                # Create LangChain Document
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)
    except Exception as e:
        logger.error(f"Failed to read or parse JSONL file {jsonl_path}: {e}")
        return False
        
    if not documents:
        logger.warning(f"No documents found in {jsonl_path}.")
        return False
        
    logger.info(f"Uploading {len(documents)} documents to the vector store...")
    
    try:
        # Upload using the generic add_documents interface of LangChain VectorStores
        vector_store.add_documents(documents)
        logger.info("Ingestion completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to upload documents: {e}")
        return False

if __name__ == "__main__":
    # Simple CLI wrapper for the ingester
    import argparse
    parser = argparse.ArgumentParser(description="Ingest gene data into Vector DB.")
    parser.add_argument("--jsonl_path", required=True, help="Path to the JSONL file containing data.")
    parser.add_argument("--index", required=True, help="Vector DB index name.")
    parser.add_argument("--vdb_provider", default="pinecone", help="Vector DB provider (e.g. pinecone).")
    parser.add_argument("--emb_provider", default="gemini", help="Embeddings provider (gemini or openai).")
    parser.add_argument("--emb_model", default="models/text-embedding-004", help="Embeddings model to use.")
    
    args = parser.parse_args()
    
    ingest_gene_data(args.jsonl_path, args.index, args.vdb_provider, args.emb_provider, args.emb_model)
