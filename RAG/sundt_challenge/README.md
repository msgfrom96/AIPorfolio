# Sundt RAG Challenge Application

## Overview

This application provides a Retrieval Augmented Generation (RAG) system for Sundt construction company data, focusing on projects and awards.

## Setup

1.  **Clone the repository.**
2.  **Navigate to `RAG/sundt_challenge` directory.**
3.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set up environment variables:**
    *   Create a file named `.env` in the `RAG/sundt_challenge` directory.
    *   Add your API keys to it. Example content:
        ```
        GOOGLE_API_KEY="your_google_api_key"
        OPENAI_API_KEY="your_openai_api_key_if_needed"
        ```

## Running the Application

1.  **Crawl Data (First time or to update data):**
    ```bash
    python run_crawler.py
    ```
    This will save JSON files into `RAG/sundt_challenge/data/raw_data/`.

2.  **Create Vector Store (First time or after crawling):**
    ```bash
    python create_vector_store.py
    ```
    This will create a FAISS index in `RAG/sundt_challenge/data/faiss_index/`.

3.  **Run the Backend Server:**
    ```bash
    uvicorn backend_app:app --reload --port 8000
    ```
    The `--reload` flag is for development and automatically reloads the server on code changes.

4.  **Access the Frontend:**
    *   Open `RAG/sundt_challenge/frontend/index.html` in your web browser.

## Project Structure

-   `backend_app.py`: Main FastAPI backend application.
-   `frontend/`: Contains HTML, CSS, and JS for the user interface.
-   `data/`: Stores crawled data (`raw_data`) and the FAISS vector index (`faiss_index`).
-   `run_crawler.py`: Script to ingest data from the Sundt website.
-   `create_vector_store.py`: Script to build the FAISS vector store.
-   `requirements.txt`: Python dependencies.
-   `.env`: API keys and other environment variables (ignored by Git).
-   `README.md`: This file. 