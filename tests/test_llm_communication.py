"""Test script to verify LLM communication works for a given config.
Handles both Gemini (supports SystemMessage) and Gemma (no SystemMessage) models."""
import os
import sys
from transcribe.config import logger
from transcribe.core.llm_factory import LLMFactory
import yaml
from langchain_core.messages import HumanMessage

def test_llm_communication(config_path):
    from transcribe.config import setup_logging
    setup_logging()
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    provider = config.get("provider", "gemini")
    models = config.get("models", ["gemini-2.5-flash"])
    if not models:
        logger.error("No models defined in config.")
        return
        
    for model_name in models:
        logger.info(f"Testing provider '{provider}' with model '{model_name}'...")
        try:
            llm = LLMFactory.get_provider(provider).get_llm(model_name, temperature=0.1)
            
            # For Gemma models, avoid SystemMessage (they don't support developer instructions).
            # Merge the instruction into the user message instead, matching GemmaAgentBuilder behavior.
            is_gemma = "gemma" in model_name.lower()
            
            if is_gemma:
                usr_msg = HumanMessage(content="You are a helpful assistant. Say 'Hello, communication is working!' and nothing else.")
                messages = [usr_msg]
            else:
                from langchain_core.messages import SystemMessage
                sys_msg = SystemMessage(content="You are a helpful assistant.")
                usr_msg = HumanMessage(content="Say 'Hello, communication is working!' and nothing else.")
                messages = [sys_msg, usr_msg]
            
            logger.info("Sending request to LLM...")
            response = llm.invoke(messages)
            
            logger.info(f"Response received successfully: {response.content}")
        except Exception as e:
            logger.error(f"Error communicating with LLM '{model_name}': {e}")
            
if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/eval_factorized_config.yaml"
    test_llm_communication(config_path)
