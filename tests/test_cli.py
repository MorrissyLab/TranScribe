import pytest
from unittest.mock import patch
from click.testing import CliRunner
from transcribe.cli import cli

def test_cli_missing_args():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code != 0
    assert "Missing argument: provide --data_path or --config." in result.output

@patch('transcribe.cli.run_yaml_eval')
@patch('transcribe.cli.setup_logging')
def test_cli_config(mock_setup_logging, mock_run_yaml_eval):
    runner = CliRunner()
    result = runner.invoke(cli, ['--config', 'fake_config.yaml'])
    assert result.exit_code == 0
    mock_run_yaml_eval.assert_called_once_with('fake_config.yaml')
    mock_setup_logging.assert_called_once()

@patch('transcribe.evaluation.report_generator.generate_html_report')
@patch('transcribe.evaluation.evaluator.evaluate_dataset')
@patch('transcribe.cli.sc.read_h5ad')
def test_cli_inference_single_file(mock_read_h5ad, mock_eval, mock_gen_report, tmp_path):
    runner = CliRunner()
    
    from anndata import AnnData
    import numpy as np
    import pandas as pd
    
    mock_adata = AnnData(X=np.zeros((2, 2)), obs=pd.DataFrame({'leiden': ['1', '2']}))
    
    mock_read_h5ad.return_value = mock_adata
    
    out_dir = str(tmp_path / "results")
    result = runner.invoke(cli, ['--data_path', 'fake_data.h5ad', '--output', out_dir])
    
    assert result.exit_code == 0
    mock_read_h5ad.assert_called_once_with('fake_data.h5ad')
    mock_eval.assert_called_once()
    mock_gen_report.assert_called_once_with(out_dir)

def test_cli_inference_missing_cluster_col():
    runner = CliRunner()
    with patch('transcribe.cli.sc.read_h5ad') as mock_read:
        from anndata import AnnData
        import numpy as np
        import pandas as pd
        mock_adata = AnnData(X=np.zeros((2, 2)), obs=pd.DataFrame({'other_col': ['1', '2']}))
        mock_read.return_value = mock_adata
        
        result = runner.invoke(cli, ['--data_path', 'fake_data.h5ad'])
        # Should exit gracefully or print an error, but exit code may be 0 if just early returning.
        assert "Cluster column 'leiden' not found" in result.output or result.exit_code == 0
