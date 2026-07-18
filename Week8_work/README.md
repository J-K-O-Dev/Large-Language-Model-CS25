# Week 8 Report: Full-Stack RAG Chatbot with React and Flask

This project culminates the work of the past weeks by integrating the RAG (Retrieval-Augmented Generation) pipeline into a complete, full-stack web application. It features a modern React frontend for the user interface and a powerful Python Flask backend to serve the language model and RAG components.

## 1. Project Overview

The goal of this week was to move beyond notebooks and standalone scripts to build a practical, interactive chatbot. The application allows a user to have a contextual conversation with an AI that can retrieve information from a local document store to provide informed, relevant answers.

- **Frontend**: A responsive chat interface built with React that persists conversation history.
- **AI Model**: Uses the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model for text generation, augmented with context from a local document collection.

## 2. Architecture

The application follows a standard client-server model.

### Frontend (Client)

- **Location**: `react_interface/`
- **Framework**: React.js
- **Key Features**:
  - A clean, modern chat UI with distinct bubbles for user and bot messages.
  - An input box that is disabled during model generation to prevent duplicate requests.
  - A loading indicator to show when the bot is "thinking".
  - **Persistent Chat History**: The conversation is automatically saved to the browser's `localStorage`, so it remains even after closing the tab.
  - **Contextual API Calls**: On each send, the entire chat history is sent to the backend, allowing the model to maintain context.
  - A "Clear Chat" button to easily start a new conversation.

### Backend (Server)

- **Location**: `backend/`
- **Framework**: Flask (with Flask-CORS for cross-origin requests from the React app).
- **Key Features**:
  - **`/chat` Endpoint**: A single API endpoint that accepts a `POST` request containing the conversation history.
  - **RAG Pipeline Integration**: All the Python classes from the `RAG_Pipe_Line.ipynb` notebook are integrated directly into the server.
    - `MultiFileLoader`: Loads various document types (`.pdf`, `.docx`, `.txt`, etc.) from the `data/` directory.
    - `Chunker`: Splits documents into smaller, manageable chunks.
    - `EmbeddingManager`: Uses `sentence-transformers` to convert text chunks into vector embeddings.
    - `VectorStoreManager`: Manages a persistent vector database using **ChromaDB**. The database is created on the first run and loaded on subsequent runs.
    - `RAGRetriever`: Retrieves the most relevant document chunks for a given user query.
  - **LLM Integration**: Uses the `TinyLlama-1.1B-Chat-v1.0` model from Hugging Face for generating responses.
  - **Context-Aware Prompting**: The backend constructs a detailed prompt for the LLM that includes a system message, the previous conversation history, and the context retrieved by the RAG pipeline.

## 3. How to Run

Follow these steps to get the full application running on your local machine.

### Prerequisites

- Python 3.8+ and `pip` (or `pipenv`).
- Node.js and `npm`.
- A CUDA-enabled GPU is highly recommended for running the backend models efficiently.

### Backend Setup

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```

2.  **Set up a Python virtual environment.** If using `pipenv`:
    ```bash
    # Specify your Python version
    pipenv install --python 3.10
    ```

3.  **Install the required Python packages:**
    ```bash
    # If using pipenv
    pipenv install Flask Flask-Cors torch transformers sentence-transformers chromadb "langchain-community>=0.0.29" pypdf docx2txt tiktoken

    # If using pip
    pip install Flask Flask-Cors torch transformers sentence-transformers chromadb "langchain-community>=0.0.29" pypdf docx2txt tiktoken
    ```

4.  **Prepare your data:**
    - Create a `data` directory inside the `Week8_work` folder (i.e., at the same level as `backend/` and `react_interface/`).
    - Place any documents (`.pdf`, `.txt`, `.docx`) you want the chatbot to have knowledge of inside this `data/` directory.

5.  **Run the Flask server:**
    ```bash
    # If using pipenv
    pipenv run python app.py

    # If using pip with a venv
    python app.py
    ```
    - The first time you run the server, it will process all documents in the `data/` directory, create embeddings, and build the ChromaDB vector store. This may take a few minutes depending on the number of documents. Subsequent runs will be much faster as they will load the existing database.
    - The backend will be running at `http://localhost:5000`.

### Frontend Setup

1.  **Open a new terminal** and navigate to the frontend directory:
    ```bash
    cd react_interface
    ```

2.  **Install the required npm packages:**
    ```bash
    npm install
    ```

3.  **Start the React development server:**
    ```bash
    npm start
    ```

4.  **Open the application:**
    - Your browser should automatically open a new tab with the chat interface at `http://localhost:3000`.
    - You can now start chatting with your RAG-powered bot!

---

This setup provides a robust foundation for a contextual chatbot. The separation of frontend and backend makes the system scalable and maintainable, while the RAG pipeline ensures that the model's responses are grounded in the provided knowledge base.

