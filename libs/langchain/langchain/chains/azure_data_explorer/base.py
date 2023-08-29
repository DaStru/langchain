import ast

from typing import Dict, Optional, List, Any

from langchain.pydantic_v1 import Extra, Field

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.chains import SequentialChain
from langchain.schema import BasePromptTemplate
from langchain.utilities.azure_data_explorer import AzureDataExplorerWrapper
from langchain.chains.azure_data_explorer.prompt import QUERY_CHECKER_PROMPT, TABLE_SELECTION_PROMPT, KQL_QUERY_PROMPT, ANSWER_FORMULATOR_PROMPT


class ADXTablesSelectorChain(Chain):
    """Use an LLM to select which tables from the database to use.""" 

    llm_chain: LLMChain
    prompt: BasePromptTemplate = TABLE_SELECTION_PROMPT
    adx_wrapper: AzureDataExplorerWrapper = Field(default_factory=AzureDataExplorerWrapper)
    database_name: str
    input_key: str = "input"
    output_key: str = "table_names"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        database_name: str,
        **kwargs: Any,
    ) -> "ADXTablesSelectorChain":
        llm_chain = LLMChain(llm=llm, prompt=TABLE_SELECTION_PROMPT)
        return cls(llm_chain=llm_chain, database_name=database_name, **kwargs)
    
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
      run_manager: Optional[CallbackManagerForChainRun] = None  
    ) -> Dict[str, str]:
        print("Entered tables selector chain")
        possible_table_names = str(self.adx_wrapper.get_table_names(database_names=[self.database_name])[self.database_name])
        llm_output = self.llm_chain.predict_and_parse(
            input=inputs[self.input_key], 
            table_names=possible_table_names,
            callbacks=run_manager.get_child() if run_manager else CallbackManagerForChainRun.get_noop_manager().get_child()
        )
        print(llm_output)
        return {self.output_key: str(llm_output)}

class KQLQueryGeneratorChain(Chain):
    """Use an LLM to generate a KQL query.""" 

    llm_chain: LLMChain
    prompt: BasePromptTemplate = KQL_QUERY_PROMPT
    adx_wrapper: AzureDataExplorerWrapper = Field(default_factory=AzureDataExplorerWrapper)
    database_name: str
    top_k: int = 5
    input_key_names: List[str] = ["input", "table_names"]
    output_key: str = "kql_query"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        database_name: str,
        **kwargs: Any,
    ) -> "KQLQueryGeneratorChain":
        llm_chain = LLMChain(llm=llm, prompt=KQL_QUERY_PROMPT)
        return cls(llm_chain=llm_chain, database_name=database_name, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return self.input_key_names

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]
    
    @property
    def _chain_type(self) -> str:
        return "kql_query_generator_chain"
    
    def _call(
      self,
      inputs: Dict[str, str],
      run_manager: Optional[CallbackManagerForChainRun] = None  
    ) -> Dict[str, str]:
        print("Entered query generator chain")
        table_info = self.adx_wrapper.get_table_info({self.database_name: ast.literal_eval(inputs["table_names"])})
        llm_output = self.llm_chain.predict(
            input=inputs["input"],
            table_info=table_info,
            top_k=self.top_k, 
            callbacks=run_manager.get_child() if run_manager else CallbackManagerForChainRun.get_noop_manager().get_child()
        )
        print(llm_output)
        return {self.output_key: llm_output}

class KQLQueryCheckerChain(Chain):
    """Use an LLM to check if a query is correct and execute it.""" 

    llm_chain: LLMChain
    prompt: BasePromptTemplate = QUERY_CHECKER_PROMPT
    adx_wrapper: AzureDataExplorerWrapper = Field(default_factory=AzureDataExplorerWrapper)
    database_name: str
    input_key: str = "kql_query"
    output_key: str = "kql_result"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        database_name: str,
        **kwargs: Any,
    ) -> "KQLQueryCheckerChain":
        llm_chain = LLMChain(llm=llm, prompt=QUERY_CHECKER_PROMPT)
        return cls(llm_chain=llm_chain, database_name=database_name, **kwargs)
    
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
      run_manager: Optional[CallbackManagerForChainRun] = None  
    ) -> Dict[str, str]:
        print("Entered query checker chain")
        checked_kql_query = self.llm_chain.predict(
            kql_query=inputs[self.input_key], 
            callbacks=run_manager.get_child() if run_manager else CallbackManagerForChainRun.get_noop_manager().get_child()
        )
        print(checked_kql_query)
        kql_result = self.adx_wrapper.run_query(database_name=self.database_name, query=checked_kql_query)
        print(kql_result)
        return {self.output_key: kql_result}

    
class ADXAnswerFormulatorChain(Chain):
    """Use an LLM to formulate an answer based on the query result.""" 

    llm_chain: LLMChain
    prompt: BasePromptTemplate = ANSWER_FORMULATOR_PROMPT
    input_key_names: List[str] = ["input", "kql_result"]
    output_key: str = "formulated_answer"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        **kwargs: Any,
    ) -> "ADXAnswerFormulatorChain":
        llm_chain = LLMChain(llm=llm, prompt=ANSWER_FORMULATOR_PROMPT)
        return cls(llm_chain=llm_chain, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return self.input_key_names

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]
    
    @property
    def _chain_type(self) -> str:
        return "adx_answer_formulator_chain"
    
    def _call(
      self,
      inputs: Dict[str, str],
      run_manager: Optional[CallbackManagerForChainRun] = None  
    ) -> Dict[str, str]:
        print("Entered tables answer formulator chain")
        formulated_answer = self.llm_chain.predict(
            input=inputs["input"], 
            kql_result=inputs["kql_result"],
            callbacks=run_manager.get_child() if run_manager else CallbackManagerForChainRun.get_noop_manager().get_child()
        )
        print(formulated_answer)
        return {self.output_key: str(formulated_answer)}


class ADXSequentialChain(Chain):
    """Get answers for questions regarding a database in Azure Data Explorer using LLM."""
    
    sequential_chain: SequentialChain
    input_key: str = "input"
    output_key: str = "final_answer"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True


    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        database_name: str,
        **kwargs: Any,
    ) -> "ADXSequentialChain":
        tables_selector_chain = ADXTablesSelectorChain.from_llm(llm, database_name, **kwargs)
        query_generator_chain = KQLQueryGeneratorChain.from_llm(llm, database_name, **kwargs)
        query_checker_chain = KQLQueryCheckerChain.from_llm(llm, database_name, **kwargs)
        answer_formulator_chain = ADXAnswerFormulatorChain.from_llm(llm, **kwargs)

        sequential_chain = SequentialChain(
            chains=[tables_selector_chain, query_generator_chain, query_checker_chain, answer_formulator_chain],
            input_variables=["input"],
            output_variables=["formulated_answer"]
        )
        return cls(sequential_chain=sequential_chain, **kwargs)


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
        return "adx_sequential_chain"
    
    def _call(
      self,
      inputs: Dict[str, str],
      run_manager: Optional[CallbackManagerForChainRun] = None  
    ) -> Dict[str, str]:
        final_answer = self.sequential_chain(
            {self.input_key: inputs[self.input_key]},
            callbacks=run_manager.get_child() if run_manager else CallbackManagerForChainRun.get_noop_manager().get_child()
        )
        return {self.output_key: str(final_answer)}