from typing import List, Optional

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.azure_data_explorer.tool import (
    QueryAzureDataExplorerTool,
    ListDatabaseNamesAzureDataExplorerTool,
    ListTableNamesAzureDataExplorerTool,
    GetTableInfoAzureDataExplorerTool,
    KQLQueryCheckerAzureDataExplorerTool
)

from langchain.utilities.azure_data_explorer import AzureDataExplorerWrapper

class AzureDataExplorerToolkit(BaseToolkit):
    """Azure Data Explorer Toolkit"""

    tools: List[BaseTool] = []

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, verbose: Optional[bool] = None) -> "AzureDataExplorerToolkit":
        tools = [
            QueryAzureDataExplorerTool(),
            ListDatabaseNamesAzureDataExplorerTool(),
            ListTableNamesAzureDataExplorerTool(),
            GetTableInfoAzureDataExplorerTool(),
        ]
        if verbose is not None:
            tools.append(KQLQueryCheckerAzureDataExplorerTool(llm=llm, verbose=verbose))
        else:
            tools.append(KQLQueryCheckerAzureDataExplorerTool(llm=llm))
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        return self.tools