"""
RAG utility functions for course retrieval using ChromaDB and OpenAI embeddings.
"""

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os
from utils.getters import load_and_clean_data
from utils.getters import clean_query
import google.generativeai as genai
from google.genai.types import GenerateContentConfig

load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
ENDPOINT = os.getenv("ENDPOINT", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-001")
QUERY_MODEL = os.getenv("QUERY_MODEL", "google/gemini-2.0-flash-001")


class Embedder(EmbeddingFunction):
    """
    Chroma-compatible embedding function that calls OpenRouter's /embeddings API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = EMBEDDING_MODEL,
    ):
        """
        Initialize the embedder.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the embeddings endpoint
            model: Model name to use for embeddings
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        if not self.api_key:
            raise RuntimeError("Embedder: API_KEY not set.")

    def __call__(self, inputs: Documents) -> Embeddings:
        """
        Generate embeddings for the input documents.

        Args:
            inputs: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not inputs:
            return []

        resp = requests.post(
            f"{self.base_url.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model,
                "input": list(inputs),
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


def generate_chunks_from_dataframe(
    df: pd.DataFrame, id_prefix: str = "dandori"
) -> List[Dict[str, Any]]:
    """
    Generate text chunks from a course dataframe.

    Args:
        df: DataFrame containing course data with columns:
            course_name, instructor, course_type, location, cost,
            learning_objectives, provided_materials, skills_developed, course_description
        id_prefix: Prefix for chunk IDs

    Returns:
        List of chunk dictionaries with id, text, and metadata
    """
    chunks = []

    for i, course in df.iterrows():
        name = course.get("course_name", "")
        instructor = course.get("instructor", "")
        course_type = course.get("course_type", "")
        location = course.get("location", "")
        cost = course.get("cost", "")
        learning_objectives = course.get("learning_objectives", "")
        materials = course.get("provided_materials", "")
        skills = course.get("skills_developed", "")
        description = course.get("course_description", "")

        # Text that will be embedded
        text = (
            f"Name: {name}\n"
            f"Instructor: {instructor}\n"
            f"Course Type: {course_type}\n"
            f"Location: {location}\n"
            f"Cost: {cost}\n"
            f"Learning Objectives: {learning_objectives}\n"
            f"Provided Materials: {materials}\n"
            f"Skills Developed: {skills}\n"
            f"Description: {description}"
        )

        # Clean up the data for filtering
        chunk = {
            "id": f"{id_prefix}_{i}_{name}",
            "text": text,
            "metadata": {
                "course_name": name,
                "instructor": instructor,
                "course_type": course_type,
                "location": location,
                "cost": float(cost) if cost else None,
                "source": f"{id_prefix}_db",
            },
        }
        chunks.append(chunk)

    return chunks


def add_chunks_to_collection(
    collection: chromadb.Collection,
    chunks: List[Dict[str, Any]],
    batch_size: int = 50,
    verbose: bool = False,
) -> None:
    """
    Add chunks to a ChromaDB collection in batches.

    Args:
        collection: ChromaDB collection to add chunks to
        chunks: List of chunk dictionaries with id, text, and metadata
        batch_size: Number of chunks to add per batch
        verbose: Whether to print progress messages
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c.get("id") for c in batch],
            documents=[c.get("text") for c in batch],
            metadatas=[c.get("metadata") for c in batch],
        )
        if verbose:
            print(f"Added batch {i // batch_size + 1} ({len(batch)} chunks)")


def create_collection(
    collection_name: str,
    api_key: str,
    base_url: str,
    model: str = EMBEDDING_MODEL,
    client=None,
) -> chromadb.Collection:
    """
    Create or get a ChromaDB collection with the specified embedding function.
    Since the collection exists in the repository, this shouldn't ever actually be run

    Args:
        collection_name: Name of the collection
        api_key: API key for embeddings
        base_url: Base URL for embeddings endpoint
        model: Model name for embeddings
        client: Optional ChromaDB client (creates new one if not provided)

    Returns:
        ChromaDB collection object
    """
    if client is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # print("-" * 50)
    # print("I'm running when I shouldn't!")
    # print("-" * 50)

    embedder = Embedder(api_key=api_key, base_url=base_url, model=model)

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedder,
    )

    return collection


def get_collection(
    collection_name: str,
):
    try:
        # FIrst try getting the collection
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        # We use both the name and embdding function to get it or it will
        # look for one with the default embedding function
        embedder = Embedder(
            api_key=OPENROUTER_API_KEY,
            base_url=ENDPOINT,
            model=EMBEDDING_MODEL,
        )
        collection = client.get_collection(
            name=collection_name, embedding_function=embedder
        )
    except:
        # If the collection doesn't exist, make one and embed pdf data
        collection = create_collection(
            collection_name=collection_name,
            api_key=OPENROUTER_API_KEY,
            base_url=ENDPOINT,
        )
        df = load_and_clean_data()
        chunks = generate_chunks_from_dataframe(df=df)
        add_chunks_to_collection(collection=collection, chunks=chunks)
    return collection


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 10,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query a ChromaDB collection for similar documents.

    Args:
        collection: ChromaDB collection to query
        query_text: Text query to search for
        n_results: Number of results to return
        where: Optional metadata filter
        where_document: Optional document content filter

    Returns:
        Dictionary containing documents, distances, metadatas, and ids
    """
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )

    return results


def format_query_results(
    results: Dict[str, Any], include_distances: bool = True
) -> List[Dict[str, Any]]:
    """
    Format collection query results into a more readable structure.

    Args:
        results: Raw results from collection.query()
        include_distances: Whether to include distance scores

    Returns:
        List of formatted result dictionaries
    """
    formatted = []

    if not results.get("documents") or not results["documents"][0]:
        return formatted

    for i in range(len(results["documents"][0])):
        result = {
            "document": results["documents"][0][i],
            "metadata": (
                results["metadatas"][0][i] if results.get("metadatas") else None
            ),
            "id": results["ids"][0][i] if results.get("ids") else None,
        }

        if include_distances and results.get("distances"):
            result["distance"] = results["distances"][0][i]

        formatted.append(result)

    return formatted


def rewrite_query_openai(
    user_question: str,
    chat_history: list[dict] = [],
) -> str:
    prompt = f"""
    Based on the chat history, rewrite the user's latest question 
    into a standalone search query that captures all necessary context.

    LATEST QUESTION:
    {user_question}
    
    STANDALONE QUERY:"""

    chat_history.append({"role": "user", "content": prompt})

    chat_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=ENDPOINT,
    )

    # Build request parameters
    request_params = {"model": "google/gemini-2.5-flash", "messages": chat_history}

    # Make the LLM request
    response = chat_client.chat.completions.create(**request_params)

    chat_client.close()

    # print("-" * 50)
    # print(f"Rewrite query:\n{prompt}")
    # print("-" * 50)
    # print("Rewritten:\n{new}".format(new=response.choices[0].message.content))
    # print("-" * 50)

    return response.choices[0].message.content


def rewrite_query_gemini(user_question, chat_history):
    # Use a cheap/fast model for this step
    rewriter_model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
    Based on the following chat history, rewrite the user's latest question 
    into a standalone search query that captures all necessary context.
    
    CHAT HISTORY:
    {chat_history}
    
    LATEST QUESTION:
    {user_question}
    
    STANDALONE QUERY:"""

    response = rewriter_model.generate_content(prompt)
    return response.text


def query_gemini_with_rag(
    chat: genai.ChatSession,
    collection_name: str,
    query: str,
    n_results: int = 5,
    collection: Optional[chromadb.Collection] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> str:  #

    try:
        if collection is None:
            collection = get_collection(collection_name=collection_name)

        history = chat.history
        search_query = rewrite_query_gemini(query, history)

        # cleaned_query = clean_query(query)
        # clean_prompt = cleaned_query["cleaned_query"]
        # constraints = cleaned_query["constraints"]

        # # Filter the contrainsts so that we only pass values and not null values
        # filtered_constraints = {key: value for key, value in constraints.items() if value is not None}

        # # Only pass the contrainsts is we have constraints
        # where_clause = filtered_constraints if filtered_constraints else None

        # Retrieve relevant documents from the collection
        rag_results = collection.query(
            query_texts=[search_query],
            n_results=n_results,
            # where=where_clause,
            where_document=where_document,
        )

        prompt = f"""
        DATA CONTEXT:
        {rag_results['documents']}
        
        USER QUESTION:
        {query}"""

        response = chat.send_message(
            content=prompt,
        )

        return response.text

    except Exception as e:
        # If it fails, we want to see the specific error
        return f"Dandori Error: {str(e)}"


def query_llm_with_rag(
    chat_client: OpenAI,
    collection_name: str,
    query: str,
    history: Optional[list[dict]] = None,
    max_history_messages: int = 5,
    model: str = QUERY_MODEL,
    n_results: int = 5,
    collection: Optional[chromadb.Collection] = None,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Query an LLM with RAG context from the collection.

    This function retrieves relevant documents from the collection and includes
    them as context in the LLM query.

    Args:
        chat_client: OpenAI client instance for making LLM requests
        collection_name: ChromaDB collection name to query for context
        query: User's query text
        history: List containing pevious messages, each a dictionary
        max_history_messages: Number of previous messages to include in query
        model: LLM model to use for generation
        n_results: Number of documents to retrieve for context
        collection: Optional ChromaDB collection to query for context, will be used instead of found with name
        system_prompt: Optional system prompt to guide the LLM's behavior
        temperature: Optional temperature for response randomness (0.0-2.0)
        max_tokens: Optional maximum tokens in the response
        where: Optional metadata filter for document retrieval
        where_document: Optional document content filter for retrieval

    Returns:
        The LLM's response as a string
    """

    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add limited history (last n messages only)
    if history:
        # Get the last N messages (exclude user query)
        recent_history = (
            history[-max_history_messages - 1 : -1]
            if len(history) > max_history_messages
            else history
        )
        messages.extend(recent_history)

    search_query = rewrite_query_openai(
        user_question=query, chat_history=list(recent_history)
    )

    if collection is None:
        collection = get_collection(collection_name=collection_name)

    # Retrieve relevant documents from the collection
    rag_results = collection.query(
        query_texts=[search_query],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )

    # Format the embedded context with the user query
    embedded_query = (
        f"DATA CONTEXT:\n{rag_results['documents']}\n\nUSER QUESTION:\n{query}"
    )

    # Add current query
    messages.append({"role": "user", "content": embedded_query})

    # Build request parameters
    request_params = {"model": model, "messages": messages}

    if temperature is not None:
        request_params["temperature"] = temperature
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens

    # Make the LLM request
    response = chat_client.chat.completions.create(**request_params)

    # print("-" * 50)
    # print(f"Full query:\n{embedded_query}")
    # print("-" * 50)
    # print("Response:\n{new}".format(new=response.choices[0].message.content))
    # print("-" * 50)

    return response.choices[0].message.content


def query_llm_with_formatted_rag(
    chat_client: OpenAI,
    collection_name: str,
    query: str,
    history: Optional[list[dict]] = None,
    max_history_messages: int = 6,
    model: str = QUERY_MODEL,
    n_results: int = 10,
    collection: Optional[chromadb.Collection] = None,
    system_prompt: Optional[str] = None,
    context_template: str = "Context:\n{context}\n\nQuestion: {query}",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Query an LLM with RAG context using a custom formatting template.

    This function provides more control over how the context is formatted
    and returns both the response and the retrieved documents.

    Args:
        chat_client: OpenAI client instance for making LLM requests
        collection_name: ChromaDB collection name to query for context
        query: User's query text
        history: List containing pevious messages with each a dictionary
        max_history_messages: Number of previous messages to include in query
        model: LLM model to use for generation
        n_results: Number of documents to retrieve for context
        collection: Optional ChromaDB collection to query for context, will be used instead of found with collection_name
        system_prompt: Optional system prompt to guide the LLM's behavior
        context_template: Template string with {context} and {query} placeholders
        temperature: Optional temperature for response randomness (0.0-2.0)
        max_tokens: Optional maximum tokens in the response
        where: Optional metadata filter for document retrieval
        where_document: Optional document content filter for retrieval

    Returns:
        Dictionary containing:
            - response: The LLM's response text
            - documents: Retrieved documents used as context
            - distances: Similarity distances for retrieved documents
            - metadatas: Metadata for retrieved documents
    """

    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Add limited history (last n messages only)
    if history:
        # Get the last N messages (exclude user query)
        recent_history = (
            history[-max_history_messages - 1 : -1]
            if len(history) > max_history_messages
            else history
        )
        messages.extend(recent_history)

    search_query = rewrite_query_openai(
        user_question=query, chat_history=list(recent_history)
    )

    if collection is None:
        collection = get_collection(collection_name=collection_name)

    # Retrieve relevant documents from the collection
    rag_results = collection.query(
        query_texts=[search_query],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )

    # Format context from retrieved documents
    context_parts = []
    for i, doc in enumerate(rag_results["documents"][0]):
        context_parts.append(f"[Document {i+1}]\n{doc}")
    context = "\n\n".join(context_parts)

    # Apply the template
    formatted_prompt = context_template.format(context=context, query=query)

    # Add current query
    messages[-1] = {"role": "user", "content": formatted_prompt}

    # Build request parameters
    request_params = {"model": model, "messages": messages}

    if temperature is not None:
        request_params["temperature"] = temperature
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens

    # Make the LLM request
    response = chat_client.chat.completions.create(**request_params)

    return {
        "response": response.choices[0].message.content,
        "documents": rag_results["documents"][0],
        "distances": rag_results.get("distances", [[]])[0],
        "metadatas": rag_results.get("metadatas", [[]])[0],
    }
