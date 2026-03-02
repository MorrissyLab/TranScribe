from typing import Any, Type, Union
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from transcribe.core.llm_factory import LLMFactory

class BaseAgentBuilder:
    """Base class for building model-specific agent chains."""
    def __init__(self, provider: str, model_name: str, temperature: float):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.llm = LLMFactory.get_provider(provider).get_llm(model_name, temperature)

    def build_structured_chain(self, system_prompt: str, user_prompt: str, output_schema: Type[BaseModel]) -> Any:
        raise NotImplementedError

    def build_string_chain(self, system_prompt: str, user_prompt: str) -> Any:
        raise NotImplementedError

class GeminiAgentBuilder(BaseAgentBuilder):
    """Builder for high-capability models (Gemini/OpenAI) using native structural outputs."""
    def build_structured_chain(self, system_prompt: str, user_prompt: str, output_schema: Type[BaseModel]) -> Any:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        # Use native tool-calling / structured output
        return prompt | self.llm.with_structured_output(output_schema)

    def build_string_chain(self, system_prompt: str, user_prompt: str) -> Any:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        return prompt | self.llm | StrOutputParser()

class GemmaAgentBuilder(BaseAgentBuilder):
    """Builder for models with limited instructions/tool-calling (Gemma) using Pydantic parsers."""
    def build_structured_chain(self, system_prompt: str, user_prompt: str, output_schema: Type[BaseModel]) -> Any:
        parser = PydanticOutputParser(pydantic_object=output_schema)
        
        # Merge system and user messages into one "user" block for Gemma compat
        # Use a placeholder for format instructions to avoid brace interpolation errors
        combined_template = f"{system_prompt}\n\nFormat Instructions:\n{{format_instructions}}\n\n{user_prompt}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("user", combined_template)
        ])
        return prompt.partial(format_instructions=parser.get_format_instructions()) | self.llm | parser

    def build_string_chain(self, system_prompt: str, user_prompt: str) -> Any:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        prompt = ChatPromptTemplate.from_messages([
            ("user", combined_prompt)
        ])
        return prompt | self.llm | StrOutputParser()

def get_agent_builder(provider: str, model_name: str, temperature: float = 0.1) -> BaseAgentBuilder:
    """Factory to return the appropriate builder based on model name."""
    m_name = model_name.lower()
    # Explicitly check for Gemma models
    if "gemma" in m_name:
        return GemmaAgentBuilder(provider, model_name, temperature)
    # Default to Gemini/Standard behavior
    return GeminiAgentBuilder(provider, model_name, temperature)
