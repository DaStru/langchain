from typing import Dict, Optional, List, Any

from langchain.pydantic_v1 import Extra, root_validator

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.schema import BasePromptTemplate
from langchain.chains.azure_data_explorer.prompt import QUERY_CHECKER_PROMPT

class KQLQueryCheckerChain(Chain):
    """Use an LLM to check if a query is correct.""" 

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = QUERY_CHECKER_PROMPT
    input_key: str = "query"
    output_key: str = "corrected_query"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True


    @root_validator()
    def initialize_llm_chain(cls, values: Dict) -> Dict:
        if "llm_chain" not in values and values["llm"] is not None:
            values["llm_chain"] = LLMChain(
                llm = values.get("llm"),
                prompt = values.get("prompt", QUERY_CHECKER_PROMPT)
            )
        if values["llm_chain"].prompt.input_variables != ["query"]:
            raise ValueError(
                "LLM chain for KQLQueryCheckerTool must have input variable ['query']"
            )
        return values
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = QUERY_CHECKER_PROMPT,
        **kwargs: Any,
    ) -> "KQLQueryCheckerChain":
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]
    
    @property
    def _chain_type(self) -> str:
        return "kql_query_checker_chain"
    
    def _call(
      self,
      inputs: Dict[str, str],
      run_manager: Optional[CallbackManagerForToolRun] = None  
    ) -> Dict[str, str]:
        llm_output = self.llm_chain.predict(
            query=inputs[self.input_key], 
            callbacks=run_manager.get_child() if run_manager else None
        )
        return {self.output_key: llm_output}
    
    
class ADXTablesSelectorChain(Chain):
    """Use an LLM to select which tables from the database to use.""" 

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    prompt: BasePromptTemplate = QUERY_CHECKER_PROMPT
    input_key: str = "query"
    output_key: str = "answer"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True


    @root_validator()
    def initialize_llm_chain(cls, values: Dict) -> Dict:
        if "llm_chain" not in values and values["llm"] is not None:
            values["llm_chain"] = LLMChain(
                llm = values.get("llm"),
                prompt = values.get("prompt", QUERY_CHECKER_PROMPT)
            )
        if values["llm_chain"].prompt.input_variables != ["query"]:
            raise ValueError(
                "LLM chain for KQLQueryCheckerTool must have input variable ['query']"
            )
        return values
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = QUERY_CHECKER_PROMPT,
        **kwargs: Any,
    ) -> "KQLQueryCheckerChain":
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]
    
    @property
    def _chain_type(self) -> str:
        return "adx_tables_selector_chain"
    
    def _call(
      self,
      inputs: Dict[str, str],
      run_manager: Optional[CallbackManagerForToolRun] = None  
    ) -> Dict[str, str]:
        llm_output = self.llm_chain.predict(
            query=inputs[self.input_key], 
            callbacks=run_manager.get_child() if run_manager else None
        )
        return {self.output_key: llm_output}
    