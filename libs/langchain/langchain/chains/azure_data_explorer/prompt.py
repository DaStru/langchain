from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers.list import CommaSeparatedListOutputParser

_KQL_QUERY_TEMPLATE = """You are a Kuso Query Language (KQL) expert. Given an input question, create a syntactically correct KQL (not SQL!!) query to run in Azure Data Explorer.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the limit clause as per KQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use now() function to get the current date, if the question involves "today".
Don't forget to put the chosen table at the beginning of the query.

Only use the following tables:
{table_info}

Question: {input}

KQL query: """

KQL_QUERY_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"],
    template=_KQL_QUERY_TEMPLATE
)


_QUERY_CHECKER_TEMPLATE = """
{kql_query}

The Kusto Query Language (KQL) query above generated the following error:
{query_error}

Double check the Kusto Query Language (KQL) query for common mistakes, including:
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Missing pipe at the beginnning of a line
- SQL syntax

If there are any of the above mistakes, rewrite the query, but only if you are really sure about the mistakes. 
If there are no mistakes, just reproduce the original query without removing the table at the beginning of the query. 

Output the final Kusto Query Language (NOT SQL!!) query only.

Don't remove the table at the beginning of the query!

KQL Query: """

QUERY_CHECKER_PROMPT = PromptTemplate(
    input_variables = ["kql_query", "query_error"], 
    template = _QUERY_CHECKER_TEMPLATE
)

_TABLES_SELECTION_TEMPLATE = """
Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.

Question: {input}

Table Names: {table_names}

Relevant Table Names:"""

TABLE_SELECTION_PROMPT = PromptTemplate(
    input_variables= ["input", "table_names"],
    template=_TABLES_SELECTION_TEMPLATE,
    output_parser=CommaSeparatedListOutputParser()
)

_ANSWER_FORMULATOR_TEMPLATE = """You are a KQL expert. Given the initial question of a user and the result of the execution of a derived KQL query, formulate a appropriate answer.
Try to include all the information that is present in the result in the answer.

Initial user question: {input}

KQL query result: {kql_result}

Answer: """

ANSWER_FORMULATOR_PROMPT = PromptTemplate(
    input_variables=["input", "kql_result"],
    template=_ANSWER_FORMULATOR_TEMPLATE
)

_PLOT_GENERATOR_TEMPLATE ="""You are a python expert. Given the initial question of a user and the result of the execution of a derived KQL query, generate python code to plot the results with plotly.

Classify the question of the user in one of the following cases and deduce the chart style:
- Chart for showing change over time: Column Chart, Line Chart, Box Plot
- Charts for showing part-to-whole composition: Pie Chart, Stacked Bar Chart, Stacked Area Chart
- Charts for looking at how data is distributed: Bar Chart, Histogramm, Violin Plot, Box Plot
- Charts for comparing values between groups: Dot Plot, Line Chart, Grouped Bar Chart, Funnel Chart
- Charts for observing relationships between variables: Scatter Plot, Bubble Chart, Heatmap, Dual-axis Plot
- Charts for looking at geographical data: Choropleth, Cartograms
- Charts for showing the details of many items: Table

If you think that there is no practical way to display all the information contained in the result or the question isn't inculded in the above cases, simply create code that outputs the result as a pandas table.

If necessary, for example in case of consecutive time data, adjust the order of the data of the result.

You don't have to include the result in the final code, instead put the placeholder "__insert_result_here__" where the result has to be inserted afterwards.

Here is the initial user question: {input}

Here is the result of the execution: {kql_result_data}

Output the final python code only.

Python code: """

PLOT_GENERATOR_PROMPT = PromptTemplate(
    input_variables=["input", "kql_result_data"],
    template=_PLOT_GENERATOR_TEMPLATE
)