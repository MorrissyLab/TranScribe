import pytest
import os
import yaml
from langchain_core.messages import HumanMessage, SystemMessage

@pytest.fixture
def config_path():
    """Provides the default config path for testing."""
    default_path = "configs/eval_factorized_config.yaml"
    assert os.path.exists(default_path), f"Test config file not found at {default_path}"
    return default_path

@pytest.fixture
def test_config(config_path):
    """Loads and provides the test configuration."""
    from transcribe.config import setup_logging
    from dotenv import load_dotenv
    import os
    load_dotenv()
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        
    setup_logging()
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def test_llm_factory_initialization(test_config):
    """Tests if LLM provider and models can be identified correctly."""
    from transcribe.core.llm_factory import LLMFactory
    
    provider = test_config.get("provider", "gemini")
    models = test_config.get("models", [])
    
    assert provider is not None, "Provider should not be None"
    assert len(models) > 0, "At least one model should be specified in config"
    
def test_llm_communication(test_config):
    """Tests if the LLM can respond to a basic prompt."""
    from transcribe.core.llm_factory import LLMFactory
    
    provider = test_config.get("provider", "gemini")
    models = test_config.get("models", ["gemini-2.5-flash"])
    
    for model_name in models:
        try:
            llm = LLMFactory.get_provider(provider).get_llm(model_name, temperature=0.1)
            
            # For Gemma models, avoid SystemMessage matching our factory constraint.
            is_gemma = "gemma" in model_name.lower()
            
            if is_gemma:
                usr_msg = HumanMessage(content="You are a helpful assistant. Say 'Hello, communication is working!' and nothing else.")
                messages = [usr_msg]
            else:
                sys_msg = SystemMessage(content="You are a helpful assistant.")
                usr_msg = HumanMessage(content="Say 'Hello, communication is working!' and nothing else.")
                messages = [sys_msg, usr_msg]
            
            response = llm.invoke(messages)
            
            assert response is not None
            assert hasattr(response, 'content')
            assert len(response.content) > 0
            
        except Exception as e:
            pytest.fail(f"LLM communication failed for {model_name}: {e}")
