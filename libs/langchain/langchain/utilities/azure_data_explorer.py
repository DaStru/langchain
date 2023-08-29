from typing import Any, Dict, Optional, Literal, List, Union

from langchain.pydantic_v1 import BaseModel, Extra, root_validator, Field
from langchain.utils import get_from_dict_or_env
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForToolRun

import json


class KustoQueryError(Exception):
    """Exception raised for errors while executing a kusto query."""
    def __init__(self, error) -> None:
        self.error = error
        super().__init__(self.error)


class AzureDataExplorerWrapper(BaseModel):
    """Wrapper around Azure Data Explorer API
    
    For clarification of the authentication methods take a look here: 
        - Overview: https://github.com/Azure/azure-kusto-python#authentication-methods
        - Details: https://github.com/Azure/azure-kusto-python/blob/master/azure-kusto-data/tests/sample.py
    """
    #General configurations
    sample_rows_in_table_info: Optional[int] = None

    #Azure Kusto specific configurations
    azure_authentication_method: Optional[Literal["aad_application", "aad_application_certificate", "aad_application_certificate_sni", "no_authentication", "system_assigned_msi", "user_assigned_msi", "azure_cli", "aad_username_password", "aad_device_code"]] = None
    kusto_client: Any
    kusto_managment_client: Any
    kcsb: Any
    cluster: Optional[str] = None
    KustoServiceError: Any
    #Authenticate with AAD application
    authority_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    #Authenticate with AAD application certificate
    pem_certificate_path: Optional[str] = None
    pem_certificate: Optional[str] = None
    certification_thumbprint: Optional[str] = None
    #Authenticate with AAD application certificate Subject Name & Issuer
    public_certificate_path: Optional[str] = None
    public_certificate: Optional[str] = None
    #Authenticate with User Assigned Managed Service Identity (MSI)
    user_assigned_client_id: Optional[str] = None
    #Authenticate with AAD username and password
    username: Optional[str] = None
    password: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
    
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        sample_rows_in_table_info = int(get_from_dict_or_env(values, "sample_rows_in_table_info", "SAMPLE_ROWS_IN_TABLE_INFO", default=str(3)))
        values["sample_rows_in_table_info"] = sample_rows_in_table_info
        """Validate that cluster is in your environment variable."""
        cluster = get_from_dict_or_env(values, "cluster", "CLUSTER")
        values["cluster"] = cluster

        try:
            from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
            from azure.kusto.data.exceptions import KustoServiceError
        
        except ImportError:
            raise ImportError(
                "azure-kusto-data is not installed. "
                "Please install it with `pip install azure-kusto-data`"
            )

        """Switch on selected authentication method"""
        azure_authentication_method = get_from_dict_or_env(values, "azure_authentication_method", "AZURE_AUTHENTICATION_METHOD", default="azure_cli")
        values["azure_authentication_method"] = azure_authentication_method
        if azure_authentication_method == "aad_application":
            authority_id = get_from_dict_or_env(values, "authority_id", "AUTHORITY_ID")
            values["authority_id"] = authority_id
            client_id = get_from_dict_or_env(values, "client_id", "CLIENT_ID")
            values["client_id"] = client_id
            client_secret = get_from_dict_or_env(values, "client_secret", "CLIENT_SECRET")
            values["client_secret"] = client_secret

            kcsb = KustoConnectionStringBuilder.with_aad_application_key_authentication(cluster, client_id, client_secret, authority_id)
            values["kcsb"] = kcsb
        
        elif azure_authentication_method == "aad_application_certificate":
            authority_id = get_from_dict_or_env(values, "authority_id", "AUTHORITY_ID")
            values["authority_id"] = authority_id
            client_id = get_from_dict_or_env(values, "client_id", "CLIENT_ID")
            values["client_id"] = client_id
            pem_certificate_path = get_from_dict_or_env(values, "pem_certificate_path", "PEM_CERTIFICATE_PATH")
            values["pem_certificate_path"] = pem_certificate_path
            with open(pem_certificate_path, "r") as pem_file:
                pem_certificate = pem_file.read()
                values["pem_certificate"] = pem_certificate
            certification_thumbprint = get_from_dict_or_env(values, "certification_thumbprint", "CERTIFICATION_THUMBPRINT")
            values["certification_thumbprint"] = certification_thumbprint

            kcsb = KustoConnectionStringBuilder.with_aad_application_certificate_authentication(cluster, client_id, pem_certificate, certification_thumbprint, authority_id)
            values["kcsb"] = kcsb
        
        elif azure_authentication_method == "aad_application_certificate_sni":
            authority_id = get_from_dict_or_env(values, "authority_id", "AUTHORITY_ID")
            values["authority_id"] = authority_id
            client_id = get_from_dict_or_env(values, "client_id", "CLIENT_ID")
            values["client_id"] = client_id
            pem_certificate_path = get_from_dict_or_env(values, "pem_certificate_path", "PEM_CERTIFICATE_PATH")
            values["pem_certificate_path"] = pem_certificate_path
            with open(pem_certificate_path, "r") as pem_file:
                pem_certificate = pem_file.read()
                values["pem_certificate"] = pem_certificate
            public_certificate_path = get_from_dict_or_env(values, "public_certificate_path", "PUBLIC_CERTIFICATE_PATH")
            values["public_certificate_path"] = public_certificate_path
            with open(public_certificate_path, "r") as cert_file:
                public_certificate = cert_file.read()
                values["public_certificate"] = public_certificate
            certification_thumbprint = get_from_dict_or_env(values, "certification_thumbprint", "CERTIFICATION_THUMBPRINT")
            values["certification_thumbprint"] = certification_thumbprint

            kcsb = KustoConnectionStringBuilder.with_aad_application_certificate_sni_authentication(cluster, client_id, pem_certificate, public_certificate, certification_thumbprint, authority_id)
            values["kcsb"] = kcsb
        
        elif azure_authentication_method == "no_authentication":
            kcsb = KustoConnectionStringBuilder()
            values["kcsb"] = kcsb
        
        elif azure_authentication_method == "system_assigned_msi":
            kcsb = KustoConnectionStringBuilder.with_aad_managed_service_identity_authentication(cluster)
        
        elif azure_authentication_method == "user_assigned_msi":
            user_assigned_client_id = get_from_dict_or_env(values, "user_assigned_client_id", "USER_ASSIGNED_CLIENT_ID")
            values["user_assigned_client_id"] = user_assigned_client_id
            
            kcsb = KustoConnectionStringBuilder.with_aad_managed_service_identity_authentication(cluster, client_id=user_assigned_client_id)
            values["kcsb"] = kcsb
        
        elif azure_authentication_method == "azure_cli":
            kcsb = KustoConnectionStringBuilder.with_az_cli_authentication(cluster)
            values["kcsb"] = kcsb
        
        elif azure_authentication_method == "aad_username_password":
            authority_id = get_from_dict_or_env(values, "authority_id", "AUTHORITY_ID")
            values["authority_id"] = authority_id
            username = get_from_dict_or_env(values, "username", "USERNAME")
            values["username"] = username
            password = get_from_dict_or_env(values, "password", "PASSWORD")
            values["password"] = password

            kcsb = KustoConnectionStringBuilder.with_aad_user_password_authentication(cluster, username, password, authority_id)
            values["kcsb"] = kcsb
        
        elif azure_authentication_method == "aad_device_code":
            kcsb = KustoConnectionStringBuilder.with_aad_device_authentication(cluster)
            values["kcsb"] = kcsb
        
        kusto_client = KustoClient(values.get("kcsb"))
        values["kusto_client"] = kusto_client

        values["KustoServiceError"] = KustoServiceError
    
        return values

    def _execute_kusto_query(self, query: str, database_name: Optional[str] = None) -> Union[List, str]:
        """Execute a kusto query.
        
        Returns the the table (`List`) or the error (`str`)"""
        try:
            response =  self.kusto_client.execute(database_name, query)
            return response.primary_results[0]
        except self.KustoServiceError as e:
            raise KustoQueryError(e)

    def _list_database_names(self) -> List:
        """Get a list of the names of all databases in the cluster.
    
        You must have at least AllDatabasesMonitor permissions for the specified cluster. TODO document this requirement"""
        kusto_response = self._execute_kusto_query(f".show databases")
        return [row["DatabaseName"] for row in kusto_response]
    
    def _check_database_existence(self, database_names: List[str]) -> List:
        """Check existence of given databases in cluster.
        
        Returns a list of databases not found in cluster or empty list if all databases exist."""
        all_database_names = self._list_database_names()
        missing_databases = list(set(database_names) - set(all_database_names))
        return missing_databases

    def _list_table_names(self, database_name: str) -> List:
        """Get a list of all the tables in the specified database.
        
        You must have at least Database User, Database Viewer, or Database Monitor permissions for the database TODO document this requirement"""
        kusto_response = self._execute_kusto_query(".show database schema as json with(Table=True)", database_name)
        return list(json.loads(kusto_response.to_dict()["data"][0]["DatabaseSchema"])['Databases'][database_name]['Tables'].keys())
    
    def _check_table_existence(self, database_name: str, table_names: List[str]) -> List:
        """Check existence of given tables in given database in cluster.
        
        Returns a list of tables not found in cluster or empty list if all tables exist"""
        all_table_names = self._list_table_names(database_name)
        missing_tables = list(set(table_names) - set(all_table_names))
        return missing_tables
    
    def get_database_names(self) -> str:
        """Get list of the names of all databases in the cluster as a string or the error."""
        try:
            return ",".join(self._list_database_names())
        except KustoQueryError as e:
            return f"Error {e}"

    def get_table_names(self, database_names: List[str]) -> Dict:
        """Get a list of the names of all tables in the specified databases.

        Input is a comma seperated list of databases for which the names of the tables are to be obtained, e.g.: ['database1'] or ['database1', 'database'].

        The output is returned as a dictionary according to the following schema: {{'database1': ['table1', 'table2'], 'database2': ['table3', 'table4']}}
        
        If a database does not exist, a corresponding error message is returned."""
        missing_databases = self._check_database_existence(database_names) 
        if missing_databases:
            return f"Error: {','.join(missing_databases)} do not exist in cluster. Please check again which databases exist."

        response_dict = {}
        for database_name in database_names:
            try:
                response_dict[database_name] = self._list_table_names(database_name)
            except KustoQueryError as e:
                return f"Error {e}"   
        
        return response_dict
    
    def get_table_info(self, table_inputs: Dict[str, List[str]]) -> str:
        """Get information about specified tables.

        Expects a dictionary with a list of tables as value and the corresponding databases as respective key, e.g.:
        
        `{"database1": ["table1"], "database2": ["table3", "table4"]}`

        If a database or a table does not exist, it will return a corresponding error message.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498). 
        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as demonstrated in the paper."""
        #Check all databases for existence
        database_names = list(table_inputs.keys())
        missing_databases = self._check_database_existence(database_names) 
        if missing_databases:
            return f"Error: {','.join(missing_databases)} do not exist in cluster. Please check again which databases exist."
        
        output_str = ""
        for database, tables in table_inputs.items():
            #Check all tables of database for existence
            missing_tables = self._check_table_existence(database, tables)
            if missing_tables:
                return f"Error: {','.join(missing_tables)} do not exist in database {database}. Please check again which tables exist."
            
            #Create information according to Rajkumar et al, 2022
            output_str += f"DATABASE: {database} \n\n"
            for table in tables:
                try:
                    #Add create command for table
                    output_str += str(self._execute_kusto_query(f'.show table {table} cslschema | project strcat(".create table ",TableName," (", Schema , ")")', database))
                    #Add sample rows
                    output_str += f"\n{self.sample_rows_in_table_info} sample rows: \n"
                    output_str += f"{table} | sample {self.sample_rows_in_table_info}\n"
                    output_str += str(self._execute_kusto_query(f"{table} | sample {self.sample_rows_in_table_info}", database))
                    output_str += "\n\n"
                except KustoQueryError as e:
                    return f"Error {e}"  

        return output_str
    
    def run_query(self, database_name: str, query: str) -> str:
        """Execute a kusto query for the specified database and return a string representing the result or the error"""
        try:
            kusto_response = self._execute_kusto_query(query, database_name)
            return str(kusto_response)     
        except KustoQueryError as e:
            return f"Error {e}" 