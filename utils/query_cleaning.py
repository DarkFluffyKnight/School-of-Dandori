import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")


def clean_query(user_query: str):
    """
    Cleans user query and creates meta data json for use in RAG prompt

    Args: 
        input: Query string from user

    Return:
        Cleaned string (correct spelling etc)
        json of metadata for RAG query
    """

    try:
        # Initialize OpenAI client with OpenRouter
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Define the system instruction
        system_instruction = (
            "You are a RAG Query Optimizer. Your task is to take a raw user query and "
            "output a structured JSON object for a vector database search. "
            "1. Clean: Remove filler words and fix typos. "
            "2. Expand: If terms are abbreviated or missing context, expand them. "
            "3. Extract: Identify specific entities or categories for filtering. "
            "4. For cost, use objects like {'$lte': 200} or {'$gte': 50, '$lte': 200}. "
            "5. For location exclusions, use {'$nin': ['London', 'York']}. "
            "6. For instructor lists, use {'$in': ['Instructor1', 'Instructor2']}. "
            "\n\nReturn a JSON object with these fields (use null for fields not present in query):\n"
            "- cleaned_query: string (cleaned version of query)\n"
            "- search_intent: string (what user wants to do)\n"
            "- course_name: string or null\n"
            "- instructor_exact: string or null\n"
            "- instructor_list: array of strings or null\n"
            "- course_type: string or null\n"
            "- location_exact: string or null\n"
            "- location_exclude: array of strings or null\n"
            "- cost_min: number or null\n"
            "- cost_max: number or null\n"
            "- source: string or null"
        )

        # Make API call with JSON response format
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        print(f"Extraction Error: {e}")
        return user_query


# if __name__ == "__main__":
query = "Hey, what courses do you have in Bath? Idealy I dont want to spend more than £200. Also Buzz Beekeeper is a great instructor."
print(clean_query(query))
