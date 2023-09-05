import os
import ast
import pandas as pd

from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.agents.agent_toolkits.azure_data_explorer.toolkit import AzureDataExplorerToolkit

from langchain.chains.azure_data_explorer.base import ADXSequentialChain

from fastapi import FastAPI
from fastapi.responses import FileResponse

os.environ["CLUSTER"] = "https://langchain-demo-cluster.westeurope.kusto.windows.net"
os.environ["AZURE_AUTHENTICATION_METHOD"] = "aad_application"
os.environ["AUTHORITY_ID"] = "0f00c3b8-46cb-4aa7-a420-d66069dc74d3"
os.environ["CLIENT_ID"] = "3d103497-4e75-4fd1-80ab-83b2fd911a85"
os.environ["CLIENT_SECRET"] = "6fc8Q~4kJMstHXl7Uv8I9TGYnT-XFMpzaqBnMdwW"
os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(temperature=0)

adx_sequential_chain = ADXSequentialChain.from_llm(llm=llm, database_name="retail")

app = FastAPI()

image_cnt = 0

@app.get("/")
async def root(question: str, url: str):
    global image_cnt
    response = adx_sequential_chain.run(question)
    formulated_answer = ast.literal_eval(response)["formulated_answer"]
    formulated_answer = formulated_answer.replace("\\n", "")
    adaptive_card = {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": []
    }

    adaptive_card["body"].append(
        {
            "type": "Container",
            "items": [
                {
                    "type": "TextBlock",
                    "text": "Here is the answer:",
                    "wrap": True,
                    "weight": "bolder",
                    "size": "medium"
                },
                {
                    "type": "TextBlock",
                    "text": formulated_answer,
                    "wrap": True
                }
            ]
        },
    )

    response_data = ast.literal_eval(ast.literal_eval(response)["kql_result"])["data"]
    df = pd.json_normalize(response_data)

    adaptive_card["body"].append(
        {
            "type": "Container",
            "items": [
                {
                    "type": "TextBlock",
                    "text": "The answer is based on this data:",
                    "wrap": True,
                    "weight": "bolder",
                    "size": "medium"
                },
                {
                    "type": "ColumnSet",
                    "columns": [
                        {
                            "type": "Column",
                            "width": "stretch",
                            "horizontalAlignment": "Center",
                            "spacing": "None",
                            "separator": True,
                            "items": [
                                {
                                    "type": "TextBlock",
                                    "text": col_name,
                                    "wrap": False,
                                    "separator": True,
                                    "weight": "Bolder",
                                    "size": "Medium"
                                }
                            ]+[
                                {
                                    "type": "TextBlock",
                                    "text": row[col_name],
                                    "wrap": False,
                                    "separator": True
                                } for _, row in df.iterrows()
                            ]
                        } for col_name in df.columns
                    ]
                }
            ]
        },
    )

    response_python_code = ast.literal_eval(response)["python_plot_code"].replace("__insert_result_here__", str(response_data))
    response_python_code = response_python_code.replace("fig.show()", "")
    response_python_code += f"\nfig.write_image('./images/fig{image_cnt}.jpg')"
    exec(response_python_code)

    adaptive_card["body"].append(
        {
            "type": "Container",
            "items": [
                {
                    "type": "TextBlock",
                    "text": "Here is the corresponding plot:",
                    "wrap": True,
                    "weight": "bolder",
                    "size": "medium"
                },
                {
                    "type": "Image",
                    "url": f"{url}/media/{image_cnt}"
                }
            ]
        },
    )
    image_cnt += 1

    return {"message": adaptive_card}

@app.get("/media/{image_number}")
async def return_plot(image_number):
    return FileResponse(f"./images/fig{image_number}.jpg")