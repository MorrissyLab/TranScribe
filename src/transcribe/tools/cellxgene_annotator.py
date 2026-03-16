import pandas as pd
import numpy as np
import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
from transcribe.config import logger

class CellxGeneAnnotator:
    def __init__(self, organism: str = "Human"):
        """
        Dynamic WMG-based annotator.
        Queries the official CellxGene WMG API (v2) for global expression summaries.
        """
        self.api_url = "https://api.cellxgene.cziscience.com/wmg/v2"
        self.organism_query = organism
        self.organism_id = None
        self.cell_type_map = {}
        self.gene_id_map = {}
        self.tissue_id_map = {}
        self._metadata_fetched = False
        self.metadata_lock = threading.Lock()
        logger.info(f"Initialized dynamic CellxGene WMG annotator for {organism}.")

    def _fetch_metadata(self):
        """Fetches cell type, gene, and tissue metadata from WMG filter dimensions."""
        with self.metadata_lock:
            if self._metadata_fetched:
                return

            try:
                logger.debug(f"Fetching WMG metadata for organism: {self.organism_query}...")
                resp = requests.get(f"{self.api_url}/primary_filter_dimensions")
                resp.raise_for_status()
                data = resp.json()
                
                # Helper to map common names to scientific names
                COMMON_NAMES = {
                    "human": "Homo sapiens",
                    "mouse": "Mus musculus"
                }
                search_term = COMMON_NAMES.get(self.organism_query.lower(), self.organism_query).lower()

                # Map Organism Name or ID
                organisms = data.get("organism_terms", [])
                for item in organisms:
                    for ont_id, name in item.items():
                        if search_term in name.lower() or search_term == ont_id.lower():
                            self.organism_id = ont_id
                            logger.info(f"Mapped '{self.organism_query}' to {name} ({ont_id})")
                            break
                    if self.organism_id:
                        break
                
                if not self.organism_id:
                    logger.warning(f"Organism '{self.organism_query}' not found. Defaulting to Human (NCBITaxon:9606).")
                    self.organism_id = "NCBITaxon:9606"

                # Genes mapping for selected organism
                gene_list = data.get("gene_terms", {}).get(self.organism_id, [])
                for item in gene_list:
                    for ens_id, symbol in item.items():
                        self.gene_id_map[str(symbol).upper()] = str(ens_id)
                
                # Tissue mapping for selected organism
                tissue_list = data.get("tissue_terms", {}).get(self.organism_id, [])
                for item in tissue_list:
                    for ub_id, name in item.items():
                        self.tissue_id_map[str(name).lower()] = str(ub_id)

                # Cell types mapping (fetched from /filters to ensure coverage)
                f_resp = requests.post(f"{self.api_url}/filters", json={"filter": {"organism_ontology_term_id": self.organism_id}})
                f_resp.raise_for_status()
                f_data = f_resp.json()
                ct_list = f_data.get("filter_dims", {}).get("cell_type_terms", [])
                for item in ct_list:
                    for ct_id, name in item.items():
                        self.cell_type_map[ct_id] = name

                self._metadata_fetched = True
                logger.info(f"Loaded {len(self.cell_type_map)} cell types, {len(self.gene_id_map)} genes, and {len(self.tissue_id_map)} tissues.")
            except Exception as e:
                logger.error(f"Failed to fetch WMG metadata: {e}")
                self._metadata_fetched = True

    def _query_wmg(self, gene_symbols: List[str], tissue: str = "Unknown") -> Dict[str, Any]:
        """Queries the WMG API for a list of gene symbols."""
        self._fetch_metadata()
        
        # Map symbols to Ensembl IDs
        ensembl_ids = [self.gene_id_map.get(s.upper()) for s in gene_symbols if s.upper() in self.gene_id_map]
        if not ensembl_ids:
            return {}

        # Map tissue to UBERON ID (fuzzy match)
        tissue_id = None
        t_query = tissue.lower()
        for name, ub_id in self.tissue_id_map.items():
            if t_query in name or name in t_query:
                tissue_id = ub_id
                break
        
        try:
            payload = {
                "filter": {
                    "gene_ontology_term_ids": ensembl_ids,
                    "organism_ontology_term_id": self.organism_id
                },
                "is_rollup": True
            }
            if tissue_id:
                payload["filter"]["tissue_ontology_term_ids"] = [tissue_id]

            resp = requests.post(f"{self.api_url}/query", json=payload)
            resp.raise_for_status()
            return resp.json().get("expression_summary", {})
        except Exception as e:
            logger.error(f"WMG query failed: {e}")
            return {}

    def query(self, marker_genes: List[str], tissue: str = "Unknown") -> Dict[str, Any]:
        """Queries the WMG API and returns formatted results."""
        if not marker_genes:
            return {
                "prediction": "No Genes",
                "score": 0.0,
                "candidates": []
            }

        data = self._query_wmg(marker_genes, tissue=tissue)
        if not data:
            return {
                "prediction": "Unknown",
                "score": 0.0,
                "candidates": []
            }

        # Aggregate scores per cell type across all genes
        cell_type_scores = {}
        
        for gene_id, level2_data in data.items():
            for l2_id, l3_data in level2_data.items():
                if l2_id == "tissue_stats":
                    continue
                
                # If l3_data is a dict of cell types (depth 3)
                if isinstance(l3_data, dict):
                    for ct_id, entry in l3_data.items():
                        if ct_id == "tissue_stats" or not ct_id.startswith("CL:"):
                            continue
                        
                        aggregated = entry.get("aggregated", {})
                        me = aggregated.get("me", 0.0)
                        pc = aggregated.get("pc", 0.0)
                        
                        if ct_id not in cell_type_scores:
                            cell_type_scores[ct_id] = 0.0
                        cell_type_scores[ct_id] += (me * pc)
                elif l2_id.startswith("CL:"):
                    # Depth 2 case: gene -> cell_type
                    entry = l3_data
                    aggregated = entry.get("aggregated", {})
                    me = aggregated.get("me", 0.0)
                    pc = aggregated.get("pc", 0.0)
                    
                    if l2_id not in cell_type_scores:
                        cell_type_scores[l2_id] = 0.0
                    cell_type_scores[l2_id] += (me * pc)

        # Map IDs to names and format results
        results = []
        num_hit_genes = len(data) if data else 1
        
        for ct_id, raw_score in cell_type_scores.items():
            name = self.cell_type_map.get(ct_id, ct_id)
            if not name.startswith("CL:") and name == ct_id and not ct_id.startswith("CL:"):
                continue

            norm_score = round(raw_score / num_hit_genes, 5)
            results.append((name, norm_score))

        results.sort(key=lambda x: x[1], reverse=True)
        
        if not results:
            return {
                "prediction": "Unknown",
                "score": 0.0,
                "candidates": []
            }

        top_cand, top_score = results[0]
        
        return {
            "prediction": top_cand,
            "score": top_score,
            "candidates": results[:10]  # Return top 10
        }

def process_excel_markers(excel_path: str) -> Dict[str, Any]:
    """
    Parses wide-format Excel file for markers per cluster/factor.
    """
    if not os.path.exists(excel_path):
        logger.error(f"Excel file not found: {excel_path}")
        return {}

    xls = pd.ExcelFile(excel_path)
    data_map = {}

    for sheet_name in xls.sheet_names:
        # metadata extraction
        full_df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        exp_header = str(full_df.iloc[0, 0])
        tissue = "Sarcoma" if "sarcoma" in exp_header.lower() else ("Muscle" if "muscle" in exp_header.lower() else "Unknown")

        # data parsing
        df = pd.read_excel(excel_path, sheet_name=sheet_name, skiprows=2)
        data_map[sheet_name] = {"tissue": tissue, "clusters": {}}
        
        for col in df.columns:
            if "Cluster" in str(col) or "Factor" in str(col):
                genes = df[col].dropna().astype(str).tolist()
                genes = [g.split('.')[0].strip() for g in genes if g.strip() and g.lower() != 'nan']
                data_map[sheet_name]["clusters"][str(col)] = genes

    return data_map

def run_census_annotation(excel_input: str, output_path: Optional[str] = None, organism: str = "Human", tissue: Optional[str] = None):
    """
    Simplified annotation process.
    """
    logger.info(f"Starting CellxGene annotation for {excel_input}...")
    
    excel_input_path = Path(excel_input)
    exp_data = process_excel_markers(str(excel_input_path))
    if not exp_data:
        return

    annotator = CellxGeneAnnotator(organism=organism)
    
    if output_path is None:
        output_path = excel_input_path.parent / "cellxgene_annotations.csv"
    
    csv_file = Path(output_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    
    columns = [
        "Experiment", "Cluster", "Tissue_Metadata", "Cell_Type_Prediction",
        "Score", "Candidates (scores)", "Top_Markers_Used"
    ]

    # Initialize CSV with header
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)
    
    write_lock = threading.Lock()

    def process_cluster(sheet_name, cluster_id, genes, use_tissue):
        res = annotator.query(genes, tissue=use_tissue)
        row = {
            "Experiment": sheet_name,
            "Cluster": cluster_id,
            "Tissue_Metadata": use_tissue,
            "Cell_Type_Prediction": res["prediction"],
            "Score": res["score"],
            "Candidates (scores)": str(res["candidates"]),
            "Top_Markers_Used": ", ".join(genes)
        }
        with write_lock:
            pd.DataFrame([row]).to_csv(csv_file, mode='a', header=False, index=False)
        return row

    exp_list = list(exp_data.items())
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        for sheet_name, content in tqdm(exp_list, desc="Processing Experiments"):
            use_tissue = tissue or content["tissue"]
            clusters = content["clusters"]
            
            futures = []
            for cluster_id, genes in clusters.items():
                futures.append(executor.submit(process_cluster, sheet_name, cluster_id, genes, use_tissue))
            
            for future in tqdm(futures, desc=f"Annotating {sheet_name}", leave=False):
                future.result()

    return str(csv_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_census_annotation(sys.argv[1])
