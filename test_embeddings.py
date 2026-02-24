"""
Test script to debug OpenRouter embeddings API issues
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
endpoint = os.getenv("ENDPOINT", "https://openrouter.ai/api/v1")

print(f"API Key: {api_key[:20]}..." if api_key else "No API key found")
print(f"Endpoint: {endpoint}")
print("\n" + "="*50)

# Test 1: Try with google/gemini-embedding-001
print("\nTest 1: google/gemini-embedding-001")
try:
    resp = requests.post(
        f"{endpoint.rstrip('/')}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Course RAG Test",
        },
        json={
            "model": "google/gemini-embedding-001",
            "input": ["Hello world"],
        },
        timeout=60,
    )
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Success! Embedding dimension: {len(data['data'][0]['embedding'])}")
    else:
        print(f"Error Response: {resp.text}")
except Exception as e:
    print(f"Exception: {e}")

# Test 2: Try with a different model (OpenAI's embedding model)
print("\n" + "="*50)
print("\nTest 2: openai/text-embedding-3-small")
try:
    resp = requests.post(
        f"{endpoint.rstrip('/')}/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "Course RAG Test",
        },
        json={
            "model": "openai/text-embedding-3-small",
            "input": ["Hello world"],
        },
        timeout=60,
    )
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Success! Embedding dimension: {len(data['data'][0]['embedding'])}")
    else:
        print(f"Error Response: {resp.text}")
except Exception as e:
    print(f"Exception: {e}")

# Test 3: List available models
print("\n" + "="*50)
print("\nTest 3: Checking available models")
try:
    resp = requests.get(
        f"{endpoint.rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        timeout=60,
    )
    if resp.status_code == 200:
        models = resp.json()
        embedding_models = [m for m in models.get('data', []) if 'embedding' in m.get('id', '').lower()]
        print(f"Found {len(embedding_models)} embedding models:")
        for model in embedding_models[:10]:  # Show first 10
            print(f"  - {model.get('id')}")
    else:
        print(f"Error: {resp.status_code} - {resp.text}")
except Exception as e:
    print(f"Exception: {e}")
