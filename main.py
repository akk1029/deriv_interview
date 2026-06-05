import os
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Local Extractive RAG Service")

# --- Global In-Memory State ---
# In a production app, this would be a specialized vector database or cache.
GLOBAL_INDEX = {
    "chunks": [],        # List of text strings
    "sources": [],       # List of source filenames mapping 1:1 to chunks
    "vectorizer": None,  # Fitted TfidfVectorizer instance
    "matrix": None       # Encoded TF-IDF matrix for the chunks
}

# --- Pydantic Schemas ---
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    context_status: str

# --- Helper Functions ---
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks based on words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# --- Endpoints ---
@app.post("/index", summary="Load and index text documents from the docs/ folder")
def index_documents():
    docs_dir = "docs"
    
    if not os.path.exists(docs_dir) or not os.path.isdir(docs_dir):
        raise HTTPException(status_code=404, detail=f"'{docs_dir}/' directory not found.")
        
    all_chunks = []
    all_sources = []
    doc_count = 0

    # Read and process all .txt files
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            doc_count += 1
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            chunks = chunk_text(content)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_sources.append(filename)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No text documents or content found to index.")

    # Build the TF-IDF index
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    # Save to global state
    GLOBAL_INDEX["chunks"] = all_chunks
    GLOBAL_INDEX["sources"] = all_sources
    GLOBAL_INDEX["vectorizer"] = vectorizer
    GLOBAL_INDEX["matrix"] = tfidf_matrix

    return {
        "status": "success",
        "documents_indexed": doc_count,
        "chunks_created": len(all_chunks)
    }

@app.post("/ask", response_model=AskResponse, summary="Query the indexed documents")
def ask_question(request: AskRequest):
    question = request.question.strip()
    
    # 1. Input & State Validation
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if GLOBAL_INDEX["matrix"] is None:
        raise HTTPException(status_code=400, detail="Index is empty. Please call /index first.")

    # 2. Retrieval
    vectorizer = GLOBAL_INDEX["vectorizer"]
    tfidf_matrix = GLOBAL_INDEX["matrix"]
    
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top 2 most relevant chunks
    top_indices = similarities.argsort()[::-1][:2]
    
    # 3. Guardrail Thresholding
    # If the strongest match score is very low, assume the context doesn't have the answer.
    SIMILARITY_THRESHOLD = 0.1 
    best_score = similarities[top_indices[0]] if len(top_indices) > 0 else 0.0

    if best_score < SIMILARITY_THRESHOLD:
        return AskResponse(
            answer="I'm sorry, but I cannot find any strong evidence in the provided documents to answer your question.",
            sources=[],
            context_status="insufficient_context"
        )

    # 4. Extractive Generation
    retrieved_chunks = []
    retrieved_sources = []
    
    for idx in top_indices:
        # Avoid pulling in irrelevant chunks if only one was strong
        if similarities[idx] >= SIMILARITY_THRESHOLD:
            retrieved_chunks.append(GLOBAL_INDEX["chunks"][idx])
            retrieved_sources.append(GLOBAL_INDEX["sources"][idx])

    # Synthesize answer (extractive approach combining top chunks)
    combined_answer = "\n\n".join(retrieved_chunks)
    # De-duplicate source names for cleaner presentation
    unique_sources = list(set(retrieved_sources))

    return AskResponse(
        answer=combined_answer,
        sources=unique_sources,
        context_status="answered_from_docs"
    )