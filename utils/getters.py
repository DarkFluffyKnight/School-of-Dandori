import pandas as pd
from ast import literal_eval
import streamlit as st
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

"""
Functions used for getting data. Specifically, loading it from a csv and parseing it to produce the dataframe we use for displaying the courses.
"""


def parse_list(data: str) -> list[str]:
    """
    Takes a string literal of a list of strings and outputs a sorted list of strings. Returns empty list in case of failure.

    Args:
        data (str): The string to interpret as list of strings e.g. \"[\"skill_1\", \"skill_2\", \"skill_3\"]\"

    Returns:
        list[str]: List of strings e.g. [\"skill_1\", \"skill_2\", \"skill_3\"]
    """
    try:
        return literal_eval(data)
    except:
        return []


def get_all_unique_skills(df: pd.DataFrame) -> list[str]:
    """
    Takes in dataframe and makes a list of all unique skills in the 'skills_developed' column.
    Each entry is a string literal representing a list of strings, each string a skill.

    Args:
        df (pd.DataFrame): Pandas dataframe, specialised for this project in particular

    Returns:
        list[str]: List of strings, one unique skill per string
    """
    all_skills = set()
    for skills_list in df["skills_developed"]:
        if isinstance(skills_list, list):
            all_skills.update(skills_list)
    return sorted(list(all_skills))


def get_all_instructors(df: pd.DataFrame) -> list[str]:
    """
    Takes in dataframe and makes a list of all unique instructor names in the 'instructor' column.

    Args:
        df (pd.DataFrame): Pandas dataframe

    Returns:
        list[str]: List of strings, one unique instructor name per string
    """
    return sorted(df["instructor"].unique().tolist())


def get_all_categories(df: pd.DataFrame) -> list[str]:
    """
    Takes in dataframe and makes a list of all unique course types in the 'course_type' column.

    Args:
        df (pd.DataFrame): Pandas dataframe

    Returns:
        list[str]: List of strings, one unique course type per string
    """
    return sorted(df["course_type"].unique().tolist())


def get_all_locations(df: pd.DataFrame) -> list[str]:
    """
    Takes in dataframe and makes a list of all unique locations in the 'location' column.

    Args:
        df (pd.DataFrame): Pandas dataframe

    Returns:
        list[str]: List of strings, one unique location per string
    """
    return sorted(df["location"].unique().tolist())


@st.cache_data
def load_and_clean_data(path: str = "course_data.csv") -> pd.DataFrame:
    """
    Loads data from the course_data.csv dataset and converts the prices to floats, and string representations of lists of strings to actual lists of strings. Uses streamlit's cache_data decorater to save reloading each time.

    Returns:
        pd.DataFrame: dataframe with complete, clean information about course details
    """
    df = pd.read_csv(path)
    df["cost"] = df["cost"].apply(
        lambda c: pd.to_numeric(str(c).replace("£", "").replace(",", ""))
    )
    for col in ["learning_objectives", "provided_materials", "skills_developed"]:
        df[col] = df[col].apply(parse_list)
    return df

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
            "You are a RAG Query Optimizer. Extract a cleaned query and constraints.\n\n"
            "Return JSON with this EXACT structure:\n"
            "{\n"
            '  "cleaned_query": "cleaned search terms",\n'
            '  "constraints": {\n'
            '    "course_name": string or null,\n'
            '    "instructor": string OR {"$in": [list]} OR null,\n'
            '    "course_type": string or null,\n'
            '    "location": string OR {"$nin": [list]} OR null,\n'
            '    "cost": number OR {"$gte": num} OR {"$lte": num} OR null,\n'
            '    "source": null\n'
            '  }\n'
            '}\n\n'
            "RULES:\n"
            "1. cleaned_query: Remove filler words, fix typos, expand abbreviations\n"
            "2. For 'cheap/affordable': cost = {'$lte': 80}\n"
            "3. For 'expensive/premium': cost = {'$gte': 100}\n"
            "4. For price range: cost = {'$and': [{'$gte': min}, {'$lte': max}]}\n"
            "5. For 'not in X': location = {'$nin': ['X']}\n"
            "6. For 'with X or Y': instructor = {'$in': ['X', 'Y']}\n"
            "7. Set unused fields to null\n"
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