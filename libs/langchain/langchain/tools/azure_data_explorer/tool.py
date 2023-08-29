"""Tools for interacting with Azure Data Explorer"""

from typing import Optional, Type, List, Dict

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.base import BaseTool
from langchain.schema.language_model import BaseLanguageModel
from langchain.utilities.azure_data_explorer import AzureDataExplorerWrapper
from langchain.chains.azure_data_explorer.base import KQLQueryCheckerChain

class QueryAzureDataExplorerSchema(BaseModel):
    """Input for QueryKQLDatabaseTool"""

    query: str = Field(description="detailed and correct KQL query")
    database_name: str = Field(description="name of the database that holds the table referenced in the query")

class QueryAzureDataExplorerTool(BaseTool):
    """Tool for querying an Azure Data Explorer database."""

    name: str = "kql_query_azure_data_explorer"
    description: str = """
    Input to this tool is a detailed and correct KQL query as well as the name of the corresponding database.
    Output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again."""
    args_schema: Type[QueryAzureDataExplorerSchema] = QueryAzureDataExplorerSchema
    azure_data_explorer_wrapper: AzureDataExplorerWrapper = Field(default_factory=AzureDataExplorerWrapper)

    def _run(
        self,
        query: str,
        database_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool"""
        return self.azure_data_explorer_wrapper.run_query(database_name=database_name, query=query)
    


class ListDatabaseNamesAzureDataExplorerSchema(BaseModel):
    """Input for ListDatabaseNamesAzureDataExplorerTool"""
    
    tool_input: str = Field(default="", description='Empty string ("")')

class ListDatabaseNamesAzureDataExplorerTool(BaseTool):
    """Tool for getting a list of the names of all databases in the cluster."""

    name: str = "list_database_names_azure_data_explorer"
    description: str = """
    This tool allows to get the name of all databases in the Azure Data Explorer Cluster. 
    Input is an empty string, output is a comma separated list of databases in the Azure Data Explorer cluster."""
    args_schema: Type[ListDatabaseNamesAzureDataExplorerSchema] = ListDatabaseNamesAzureDataExplorerSchema
    azure_data_explorer_wrapper: AzureDataExplorerWrapper = Field(default_factory=AzureDataExplorerWrapper)

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool"""
        return self.azure_data_explorer_wrapper.get_database_names()
    


class ListTableNamesAzureDataExplorerSchema(BaseModel):
    """Input for ListTableNamesAzureDataExplorerTool"""
    
    database_names: str = Field(description="""comma seperated list of databases for which the names of the tables are to be obtained, e.g.: "database1" or "database1, database" """)


class ListTableNamesAzureDataExplorerTool(BaseTool):
    """Tool for getting a list of the names of all tables in the specified databases."""

    name: str = "list_table_names_azure_data_explorer"
    description: str = """
    This tool allows to get the name of all tables for given databases in the Azure Data Explorer Cluster.
    Input is a comma seperated list of databases for which the names of the tables are to be obtained, e.g.: "database1" or "database1, database".
    The output is returned as a dictionary according to the following schema: {{"database1": ["table1", "table2"], "database2": ["table3", "table4"]}}
    Make sure to always provide a list of tables and always ensure that the databases exist before using this tool.
    If a database does not exist, a corresponding error message is returned. Recheck that all databases exist in the cluster before trying again."""
    args_schema: Type[ListTableNamesAzureDataExplorerSchema] = ListTableNamesAzureDataExplorerSchema
    azure_data_explorer_wrapper: AzureDataExplorerWrapper = Field(default_factory=AzureDataExplorerWrapper)

    def _run(
        self,
        database_names: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Dict:
        """Use the tool"""
        return self.azure_data_explorer_wrapper.get_table_names(database_names=[database for database in database_names.split(", ")])
    


class GetTableInfoAzureDataExplorerSchema(BaseModel):
    """Input for GetTableInfoAzureDataExplorerTool"""

    table_inputs: Dict[str, List[str]] = Field(description="""dictionary with a list of tables for which further information is wanted as value and the corresponding databases as respective key, e.g.: {"database1": ["table1"], "database2": ["table3", "table4"]} """)

class GetTableInfoAzureDataExplorerTool(BaseTool):
    """Tool for getting information about specified tables."""

    name: str = "get_table_information_azure_data_explorer"
    description: str = """
    This tool allows the get further information about specified tables.
    Make sure to always provide a list of tables!!
    Input is a dictionary with a list of tables for which further information is wanted as value and the corresponding databases as respective key, e.g.: {{"database1": ["table1"], "database2": ["table3", "table4"]}}.
    This is an example for an input: {{"database1": ["table1"], "database2": ["table3", "table4"]}}
    Output is in the form of the schema and sample rows as a string."""
    #args_schema: Type[GetTableInfoAzureDataExplorerSchema] = GetTableInfoAzureDataExplorerSchema
    azure_data_explorer_wrapper: AzureDataExplorerWrapper = Field(default_factory=AzureDataExplorerWrapper)

    def _run(
        self,
        table_inputs: Dict[str, List[str]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool"""
        return self.azure_data_explorer_wrapper.get_table_info(table_inputs=table_inputs)
    


class KQLQueryCheckerAzureDataExplorerSchema(BaseModel):
    """Input for KQLQueryCheckerAzureDataExplorerTool"""

    query: str = Field(description="KQL that has to be validated")

class KQLQueryCheckerAzureDataExplorerTool(BaseTool):
    """Use an LLM to check if a query is correct.
    
    Adapted from SQL agent (and https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/)"""

    name: str = "kql_query_checker_azure_data_explorer"
    description: str = """
    Use this tool to double check if your kql query is correct before executing it.
    Always use this tool before executing a query with kql_query_azure_data_explorer!
    Input is the KQL that has to be validated.
    Output is the corrected KQL query.
    """
    args_schema: Type[KQLQueryCheckerAzureDataExplorerSchema] = KQLQueryCheckerAzureDataExplorerSchema
    llm: BaseLanguageModel
    verbose: bool = Field(default=False)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool"""
        kql_query_checker_chain = KQLQueryCheckerChain.from_llm(llm=self.llm, verbose=self.verbose)
        return kql_query_checker_chain.run(query)
