import os
import pandas as pd
from typing import List, Optional
from transcribe.config import logger

def clean_mixed_gene_names(genes_list: List[str], genome: str) -> List[str]:
    """
    Clean gene names based on genome type by removing prefixes like 'mm10___' or 'GRCh38_'.
    """
    if genome == 'None':
        return [x.replace('mm10___', '').replace('GRCh38_', '') for x in genes_list]
    elif genome == 'mm10':
        return [x.replace('mm10___', '') for x in genes_list]
    else:
        return [x.replace('GRCh38_', '') for x in genes_list]

def list_genesets(genome: Optional[str] = None) -> List[str]:
    """List available gene sets in the data/genesets directory."""
    # Find the data/genesets directory relative to the project root
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    geneset_dir = os.path.join(parent_directory, '..', '..', 'data', 'genesets')
    
    if not os.path.isdir(geneset_dir):
        geneset_dir = os.path.join(parent_directory, 'data', 'genesets')
        
    if not os.path.isdir(geneset_dir):
        logger.warning(f"Geneset directory not found at {geneset_dir}")
        return []

    available_genesets = [f.replace(".gmt", "") for f in os.listdir(geneset_dir) if f.endswith(".gmt")]

    if genome == "mm10":
        available_genesets = [f for f in available_genesets if "mouse" in f.lower()]
    elif genome == "GRCh38":
        available_genesets = [f for f in available_genesets if "human" in f.lower()]

    return available_genesets

def read_gmt_file(gmt_path: str) -> pd.DataFrame:
    """Read a .gmt file and return as a DataFrame."""
    data = []
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 2:
                data.append(parts)
    
    if not data:
        return pd.DataFrame()
        
    max_len = max(len(x) for x in data)
    padded_data = [x + [None] * (max_len - len(x)) for x in data]
    
    df = pd.DataFrame(padded_data).T
    df.columns = df.iloc[0]
    df = df.drop([0, 1]) # Drop name and description rows
    return df
