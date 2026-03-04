import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import csv
from transcribe.config import logger

def load_factorized_data(path: str) -> pd.DataFrame:
    """
    Loads a factorized matrix (NMF/cNMF) from CSV, TSV, or TXT.
    Automatically detects delimiter and ensures orientation where rows are factors and columns are genes.
    Assumes that the number of factors is much smaller than the number of genes.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Factorized data file not found at: {path}")

    logger.info(f"Loading factorized data from {path}")
    
    # Try to sniff delimiter
    try:
        with open(path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
    except csv.Error:
        # Fallback to pandas basic detection if sniffer fails
        delimiter = None

    try:
        # Load the data
        df = pd.read_csv(path, sep=delimiter, engine='python', index_col=0)
    except Exception as e:
        logger.error(f"Failed to load {path} with delimiter {delimiter}: {e}")
        # Try without index_col
        df = pd.read_csv(path, sep=delimiter, engine='python')

    # Detect orientation based on shape. Factors << Genes.
    n_rows, n_cols = df.shape
    
    if n_rows > n_cols:
        logger.info(f"Matrix shape {df.shape} suggests factors are columns. Transposing...")
        df = df.T
    else:
        logger.info(f"Matrix shape {df.shape} suggests factors are rows. Orientation is correct.")
        
    df.index = df.index.astype(str)
        
    return df

def extract_top_factor_markers(factor_df: pd.DataFrame, factor_id: str, top_n: int = 50) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract top N genes and their weights for a specific factor.
    Returns:
        genes: List of top N gene names
        weights: Dictionary mapping gene name to its weight
    """
    try:
        # Coerce factor_id to correct type if necessary (e.g. integer or string indices might differ)
        if isinstance(factor_df.index[0], str):
            factor_id = str(factor_id)
        elif isinstance(factor_df.index[0], int):
            factor_id = int(factor_id)
            
        row = factor_df.loc[factor_id]
        
    except KeyError:
        raise ValueError(f"Factor ID '{factor_id}' not found in the index. Available factors: {factor_df.index.tolist()}")

    # Sort genes by weight in descending order
    sorted_row = row.sort_values(ascending=False)
    
    top_genes = sorted_row.head(top_n)
    
    genes = top_genes.index.tolist()
    # Ensure keys are strings and values are floats for downstream JSON serialization
    weights = {str(k): float(v) for k, v in top_genes.items()}
    
    return genes, weights

