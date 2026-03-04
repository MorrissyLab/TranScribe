import pytest
import yaml
from unittest.mock import patch, mock_open
from transcribe.evaluation.yaml_runner import run_yaml_eval

@pytest.fixture
def mock_yaml_config():
    return {
        "mode": "eval",
        "provider": "gemini",
        "models": ["fake-model-1"],
        "datasets": [
            {
                "name": "TestDataset",
                "path": "toy_data",
                "modality": "single-cell",
            }
        ],
        "output": "results/test_eval",
        "num_tries": 1
    }

@patch("transcribe.evaluation.yaml_runner.Path.exists", return_value=True)
@patch("transcribe.evaluation.report_generator.generate_html_report")
@patch("transcribe.evaluation.yaml_runner.evaluate_dataset")
@patch("transcribe.evaluation.yaml_runner.fetch_toy_dataset")
def test_run_yaml_eval_success(mock_fetch_toy, mock_evaluate_dataset, mock_generate_html_report, mock_exists, mock_yaml_config):
    # Mock fetch_toy_dataset returns
    mock_fetch_toy.return_value = ("mock_adata", "mock_c_col", "mock_t_col")
    
    yaml_str = yaml.dump(mock_yaml_config)
    
    with patch("builtins.open", mock_open(read_data=yaml_str)):
        run_yaml_eval("configs/dummy.yaml")
        
    mock_fetch_toy.assert_called_once()
    mock_evaluate_dataset.assert_called_once_with(
        adata="mock_adata",
        factorized_df=None,
        raw_data_path=None,
        data_path="toy_data",
        cluster_col="mock_c_col",
        ground_truth_col="mock_t_col",
        dataset_name="TestDataset",
        run_name="TestDataset_fake-model-1",
        provider="gemini",
        model_name="fake-model-1",
        out_dir="results/test_eval",
        organism="Human",
        tissue="Unknown",
        disease="Normal",
        num_tries=1,
        modality="single-cell",
        factorized_type="sc"
    )
    mock_generate_html_report.assert_called_once_with("results/test_eval")

@patch("transcribe.evaluation.yaml_runner.Path.exists", return_value=False)
@patch("transcribe.evaluation.yaml_runner.logger.error")
def test_run_yaml_eval_missing_file(mock_logger_error, mock_exists):
    run_yaml_eval("missing.yaml")
    mock_logger_error.assert_called_once()
    assert "YAML config not found" in mock_logger_error.call_args[0][0]
