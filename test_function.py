from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get('API_KEY')
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.moonshot.cn/v1",
)

completion = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[
    {
        "role": "system",
        "content": "You are an assistant to tourists visiting a hotel.\nYou have access to a database of items (which includes ['Food and beverages', 'Miscellaneous']) that tourists can buy, you also have access to the hotel's brochure.\nIf the tourist's question cannot be answered from the database, you can refer to the brochure.\nIf the tourist's question cannot be answered from the brochure, you can ask the tourist to ask the hotel staff.\n"
    },
    {
        "role": "user",
        "content": "Can I buy a coffee?"
    }
],
    tools=[
    {
        "type": "function",
        "function": {
            "name": "get_items",
            "description": "Get a list of items from the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "string",
                        "description": "Comma separated list of item ids to fetch",
                    },
                    "categories": {
                        "type": "string",
                        "description": "Comma separated list of item categories to fetch",
                    },
                },
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "purchase_item",
            "description": "Purchase a particular item",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The given product ID, product name is not accepted here. Please obtain the product ID from the database first.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Number of items to purchase",
                    },
                },
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_pipeline_func",
            "description": "Get information from hotel brochure",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to use in the search. Infer this from the user's message. It should be a question or a statement",
                    }
                },
                "required": ["query"],
            },
        },
    }
],
    temperature=0.3,
)

print(completion.choices[0].message)