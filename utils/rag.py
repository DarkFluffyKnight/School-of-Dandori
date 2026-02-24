"""
RAG utility functions for course retrieval using ChromaDB and OpenAI embeddings.
"""

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import requests
import pandas as pd
from typing import List, Dict, Any, Optional
from openai import OpenAI


class Embedder(EmbeddingFunction):
    """
    Chroma-compatible embedding function that calls OpenRouter's /embeddings API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "google/gemini-embedding-001",
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


def create_collection(
    collection_name: str,
    api_key: str,
    base_url: str,
    model: str = "google/gemini-embedding-001",
    client: Optional[chromadb.Client] = None,
) -> chromadb.Collection:
    """
    Create or get a ChromaDB collection with the specified embedding function.

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
        client = chromadb.Client()

    embedder = Embedder(api_key=api_key, base_url=base_url, model=model)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
    )

    return collection


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


def query_llm_with_rag(
    chat_client: OpenAI,
    collection: chromadb.Collection,
    query: str,
    model: str = "google/gemini-2.0-flash-001",
    n_results: int = 10,
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
        collection: ChromaDB collection to query for context
        query: User's query text
        model: LLM model to use for generation
        n_results: Number of documents to retrieve for context
        system_prompt: Optional system prompt to guide the LLM's behavior
        temperature: Optional temperature for response randomness (0.0-2.0)
        max_tokens: Optional maximum tokens in the response
        where: Optional metadata filter for document retrieval
        where_document: Optional document content filter for retrieval

    Returns:
        The LLM's response as a string
    """
    # Retrieve relevant documents from the collection
    rag_results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )

    # Format the embedded context with the user query
    embedded_query = f"{rag_results['documents']}\n\n{query}"

    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": embedded_query})

    # Build request parameters
    request_params = {"model": model, "messages": messages}

    if temperature is not None:
        request_params["temperature"] = temperature
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens

    # Make the LLM request
    response = chat_client.chat.completions.create(**request_params)

    return response.choices[0].message.content


def query_llm_with_formatted_rag(
    chat_client: OpenAI,
    collection: chromadb.Collection,
    query: str,
    model: str = "google/gemini-2.0-flash-001",
    n_results: int = 10,
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
        collection: ChromaDB collection to query for context
        query: User's query text
        model: LLM model to use for generation
        n_results: Number of documents to retrieve for context
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
    # Retrieve relevant documents from the collection
    rag_results = collection.query(
        query_texts=[query],
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

    # Build messages list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": formatted_prompt})

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
