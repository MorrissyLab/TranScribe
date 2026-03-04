import pytest
from unittest.mock import patch, MagicMock
from transcribe.evaluation.datasets import fetch_toy_dataset, fetch_spatial_toy_dataset

@patch('transcribe.evaluation.datasets.sc.datasets.pbmc3k_processed')
def test_fetch_toy_dataset(mock_pbmc3k):
    from anndata import AnnData
    import numpy as np
    import pandas as pd
    
    # Create mock anndata
    df = pd.DataFrame({'louvain': pd.Categorical(['A', 'B', 'B', 'C'])})
    mock_adata = AnnData(X=np.zeros((4, 4)), obs=df)
    
    mock_pbmc3k.return_value = mock_adata
    
    adata, c_col, t_col = fetch_toy_dataset()
    
    assert c_col == "blind_cluster"
    assert t_col == "louvain"
    assert "blind_cluster" in adata.obs.columns
    # Ensure it's string categorization of the original integer codes
    assert list(adata.obs["blind_cluster"]) == ["0", "1", "1", "2"]


@patch('squidpy.gr.spatial_neighbors', create=True)
@patch('squidpy.datasets.visium_hne_adata', create=True)
def test_fetch_spatial_toy_dataset(mock_visium, mock_spatial_neighbors):
    from anndata import AnnData
    import numpy as np
    import pandas as pd
    
    df = pd.DataFrame({'cluster': pd.Categorical(['X', 'Y', 'Z'])})
    mock_adata = AnnData(X=np.zeros((3, 3)), obs=df)
    
    mock_visium.return_value = mock_adata
    
    try:
        adata, c_col, t_col = fetch_spatial_toy_dataset()
    except Exception as e:
        pytest.fail(f"fetch_spatial_toy_dataset failed: {e}")
        
    assert c_col == "blind_cluster"
    assert t_col == "cluster"
    assert "blind_cluster" in adata.obs.columns
    
    # Ensure squidpy was called correctly
    mock_visium.assert_called_once()
    mock_spatial_neighbors.assert_called_once_with(mock_adata)
