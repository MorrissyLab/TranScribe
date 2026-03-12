from typing import Any, Type, Union, List, Literal, get_origin, get_args
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
        self.llm = LLMFactory.get_llm(provider, model_name, temperature)

    def build_structured_chain(self, system_prompt: str, user_prompt: str, output_schema: Type[BaseModel]) -> Any:
        raise NotImplementedError

    def build_string_chain(self, system_prompt: str, user_prompt: str) -> Any:
        raise NotImplementedError

    @staticmethod
    def is_gemma_model(model_name: str) -> bool:
        """Centralized check for Gemma model family."""
        return "gemma" in model_name.lower()

class StructuredAgentBuilder(BaseAgentBuilder):
    """Builder that uses manual JSON formatting and Pydantic parsing for all models."""
    
    def _get_example_value(self, field_type: Any) -> Any:
        from typing import get_origin, get_args, List, Literal
        origin = get_origin(field_type)
        args = get_args(field_type)
        
        # Handle Literals (use first value)
        if origin is Literal and args:
            return args[0]
        # Handle Lists
        if origin is list or origin is List:
             item_type = args[0] if args else str
             return [self._get_example_value(item_type)]
        # Handle nested Pydantic models
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            nested_fields = {}
            if hasattr(field_type, "model_fields"):
                nested_fields = field_type.model_fields
            elif hasattr(field_type, "__fields__"):
                nested_fields = field_type.__fields__
            
            res = {}
            for name, f in nested_fields.items():
                f_ann = f.annotation if hasattr(f, 'annotation') else f.type_
                res[name] = self._get_example_value(f_ann)
            
            # Special case for confidence
            if "confidence" in res:
                res["confidence"] = "high"
            return res
            
        # Default
        return "..."

    def _strip_markdown(self, text_input: Any) -> str:
        text = ""
        if hasattr(text_input, "content"):
            text = str(text_input.content)
        else:
            text = str(text_input)
        
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].strip().startswith("```"):
                lines.pop(-1)
            text = "\n".join(lines).strip()
        return text

    def build_structured_chain(self, system_prompt: str, user_prompt: str, output_schema: Type[BaseModel]) -> Any:
        parser = PydanticOutputParser(pydantic_object=output_schema)
        
        example_obj = self._get_example_value(output_schema)
            
        import json as json_builtin
        example_json = json_builtin.dumps(example_obj)
        example_json_escaped = example_json.replace("{", "{{").replace("}", "}}")

        combined_template = (
            f"Role: {system_prompt}\n\n"
            f"Data: {user_prompt}\n\n"
            f"Respond ONLY with a JSON object following this EXACT structure: {example_json_escaped}\n"
            "Do NOT include any preamble or code blocks. Respond ONLY with the raw JSON.\n\n"
            "{{format_instructions}}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("user", combined_template)
        ])
        
        from langchain_core.runnables import RunnableLambda
        return (
            prompt.partial(format_instructions=parser.get_format_instructions()) 
            | self.llm 
            | RunnableLambda(self._strip_markdown) 
            | parser
        )

    def build_string_chain(self, system_prompt: str, user_prompt: str) -> Any:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        prompt = ChatPromptTemplate.from_messages([
            ("user", combined_prompt)
        ])
        return prompt | self.llm | StrOutputParser()

class NativeStructuredAgentBuilder(BaseAgentBuilder):
    """Builder for high-capability models (Gemini/OpenAI) using native structural outputs."""
    def build_structured_chain(self, system_prompt: str, user_prompt: str, output_schema: Type[BaseModel]) -> Any:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        # Use native tool-calling / structured output. 
        # Explicitly use function_calling/tool_calling to avoid metadata issues with some providers
        method = "function_calling" if self.provider == "openai" else None
        return prompt | self.llm.with_structured_output(output_schema, method=method)

    def build_string_chain(self, system_prompt: str, user_prompt: str) -> Any:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        return prompt | self.llm | StrOutputParser()

class GemmaAgentBuilder(StructuredAgentBuilder):
    """Builder for Gemma models using manual structured prompting approach."""
    pass


def get_agent_builder(provider: str, model_name: str, temperature: float = 0.1) -> BaseAgentBuilder:
    """Factory to return the appropriate builder based on model name and provider capabilities."""
    name = model_name.lower()
    
    # Gemma always uses manual Pydantic parsing
    if BaseAgentBuilder.is_gemma_model(name):
        return GemmaAgentBuilder(provider, model_name, temperature)
        
    # Gemini and OpenAI use native structured output
    if provider.lower() in ["gemini", "openai"]:
        return NativeStructuredAgentBuilder(provider, model_name, temperature)
        
    # Fallback to structured/manual for everything else
    return StructuredAgentBuilder(provider, model_name, temperature)
