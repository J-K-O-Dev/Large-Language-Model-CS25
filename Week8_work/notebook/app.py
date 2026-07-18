# backend/app.py

# --- Setup Instructions ---
# 1. Make sure you have a Python environment (e.g., venv).
# 2. Install the required packages:
#    pip install Flask Flask-Cors torch transformers sentence-transformers chromadb langchain-community pypdf docx2txt tiktoken
# 3. Place this file in a 'backend' directory inside your 'Week8_work' folder.
# 4. Make sure your data files (PDFs, etc.) are in 'Week8_work/data/'.
# 5. Run the server from your terminal:
#    python app.py
# --------------------------

import os
import uuid
import time
# import numpy as np
import torch
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import langchain_community.document_loaders as loaders
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- 1. RAG Pipeline Components (from RAG_Pipe_Line.ipynb) ---

class MultiFileLoader:
    LOADER_MAPPING = {
        ".pdf": loaders.PyPDFLoader,
        ".docx": loaders.Docx2txtLoader,
        ".txt": loaders.TextLoader,
        ".csv": loaders.CSVLoader,
        ".json": loaders.JSONLoader,
    }

    def __init__(self, directory_path):
        if not os.path.isdir(directory_path):
            raise ValueError(f"The path '{directory_path}' is not a valid directory.")
        self.directory_path = directory_path

    def load_documents(self):
        documents = []
        print(f"Loading documents from '{self.directory_path}'...")
        for filename in os.listdir(self.directory_path):
            filepath = os.path.join(self.directory_path, filename)
            if os.path.isfile(filepath):
                _, extension = os.path.splitext(filename)
                loader_class = self.LOADER_MAPPING.get(extension.lower())
                if loader_class:
                    try:
                        print(f"-> Loading '{filename}' with {loader_class.__name__}")
                        loader = loader_class(filepath)
                        documents.extend(loader.load())
                    except Exception as e:
                        print(f"  [!] Error loading '{filename}': {e}")
                else:
                    print(f"-> Skipping unsupported file type: '{filename}'")
        print(f"Finished loading. Total documents loaded: {len(documents)}")
        return documents

class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_documents(documents)

class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"Successfully loaded embedding model '{self.model_name}' on {self.model.device}")

    def embed_documents(self, documents):
        texts = [doc.page_content for doc in documents]
        return self.model.encode(texts, show_progress_bar=False)

class VectorStoreManager:
    def __init__(self, collection_name="pdf_files", directory="../data/vector_store"):
        self.collection_name = collection_name
        self.directory = directory
        self.client = chromadb.PersistentClient(path=self.directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Number of embeddings must match the number of documents")
        print(f"Adding {len(documents)} documents to vector store...")
        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(documents))]
        metadatas = [doc.metadata for doc in documents]
        doc_texts = [doc.page_content for doc in documents]
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=doc_texts)
        print("Successfully added documents to the vector store.")

class RAGRetriever:
    def __init__(self, vector_collection, embedding_manager):
        self.collection = vector_collection
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k=3):
        print(f"\n--- Retrieving context for query: '{query}' ---")
        if not query.strip():
            return []
        query_embedding = self.embedding_manager.embed_documents([Document(page_content=query)])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        retrieved_docs = [{"content": doc, "metadata": meta} for doc, meta in zip(results['documents'][0], results['metadatas'][0])]
        print(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs

# --- 2. Global Initializations ---

print("Initializing models and RAG pipeline...")

# Load LLM and Tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 # Use bfloat16 for better performance if available
)
print("LLM loaded successfully.")

# Initialize RAG components
embedding_manager = EmbeddingManager()
vectorstore = VectorStoreManager()
rag_retriever = RAGRetriever(vectorstore.collection, embedding_manager)

def setup_rag_pipeline():
    """Checks if the vector store is populated and runs the ingestion pipeline if not."""
    if vectorstore.collection.count() == 0:
        print("Vector store is empty. Starting data ingestion pipeline...")
        # The relative path should work if backend/ is inside Week8_work/
        data_path = "../data"
        if not os.path.exists(data_path):
            print(f"[ERROR] Data directory not found at '{data_path}'. Please create it and add your documents.")
            return

        loader = MultiFileLoader(directory_path=data_path)
        documents = loader.load_documents()

        if documents:
            chunker = Chunker()
            chunked_documents = chunker.chunk_documents(documents)
            chunk_embeddings = embedding_manager.embed_documents(chunked_documents)
            vectorstore.add_documents(documents=chunked_documents, embeddings=chunk_embeddings)
        else:
            print("No documents were loaded. RAG will not have context.")
    else:
        print(f"Vector store already contains {vectorstore.collection.count()} documents.")


# --- 3. Flask Application ---

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles the chat request from the frontend.
    """
    try:
        data = request.get_json()
        history = data.get('history', [])

        if not history:
            return jsonify({"error": "Chat history is empty."}), 400

        # Extract the last user message for RAG retrieval
        user_prompt = history[-1]['text']

        # Retrieve context from the vector store
        retrieved_docs = rag_retriever.retrieve(user_prompt, top_k=3)
        context_str = "\n\n".join([doc['content'] for doc in retrieved_docs])

        # Prepare the messages for the chat model
        # We will augment the last user message with the retrieved context
        messages_for_model = []
        system_prompt = "You are a helpful assistant created by Mr Jithin K. Answer the user's question based on the provided context."
        messages_for_model.append({"role": "system", "content": system_prompt})

        # Add previous conversation history
        for message in history[:-1]:
            role = "user" if message['sender'] == 'user' else "assistant"
            messages_for_model.append({"role": role, "content": message['text']})

        # Add the augmented user prompt
        augmented_prompt = f"--- CONTEXT ---\n{context_str}\n\n--- QUESTION ---\n{user_prompt}"
        messages_for_model.append({"role": "user", "content": augmented_prompt})

        # Apply the chat template
        prompt_text = tokenizer.apply_chat_template(
            messages_for_model,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # Generate response
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )
        end_time = time.perf_counter()

        # Decode the generated text
        response_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        print(f"Generated response in {end_time - start_time:.2f} seconds.")
        print(f"Response: {response_text}")

        return jsonify({"reply": response_text})

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == '__main__':
    # Run the data ingestion pipeline on startup if needed
    setup_rag_pipeline()
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)