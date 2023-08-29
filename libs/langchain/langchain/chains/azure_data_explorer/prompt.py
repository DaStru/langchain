from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers.list import CommaSeparatedListOutputParser

_QUERY_CHECKER_TEMPLATE = """
{query}
Double check the query KQL above for common mistakes, including:
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final KQL query only.

KQL Query: """

QUERY_CHECKER_PROMPT = PromptTemplate(
    input_variables = ["query"], 
    template = _QUERY_CHECKER_TEMPLATE
)

_TABLES_SELECTION_TEMPLATE = """
Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.

Question: {query}

Table Names: {table_names}

Relevant Table Names:"""

TABLE_SELECTION_PROMPT = PromptTemplate(
    input_variables= ["query", "table_names"],
    template=_TABLES_SELECTION_TEMPLATE,
    output_parser=CommaSeparatedListOutputParser()
)