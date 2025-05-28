# Import necessary libraries for RAG
from typing import Optional, Dict, List, Any
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
import json
from datetime import datetime
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# This script will create the vector store from crawled data.

# --- Data Loading ---
# Load crawled data from the most recent JSON files in the 'data' directory

# Get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "faiss_index")

def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Finds the path to the latest file matching a pattern in a directory."""
    # Construct the full pattern for glob
    search_pattern = os.path.join(directory, pattern)
    # List all files matching the pattern
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None
    # Sort files by modification time (most recent first)
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def load_data_from_latest_files(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads projects and awards data from the latest JSON files."""
    latest_projects_file = find_latest_file(directory, "sundt_projects_*.json")
    latest_awards_file = find_latest_file(directory, "sundt_awards_*.json")

    projects_data = []
    awards_data = []

    if latest_projects_file:
        print(f"Loading projects from: {latest_projects_file}")
        try:
            with open(latest_projects_file, 'r', encoding='utf-8') as f:
                projects_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Projects file not found at {latest_projects_file}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {latest_projects_file}")
        except Exception as e:
            print(f"An unexpected error occurred loading projects: {e}")
    else:
        print(f"Warning: No projects file found matching 'sundt_projects_*.json' in '{directory}'")

    if latest_awards_file:
        print(f"Loading awards from: {latest_awards_file}")
        try:
            with open(latest_awards_file, 'r', encoding='utf-8') as f:
                awards_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Awards file not found at {latest_awards_file}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {latest_awards_file}")
        except Exception as e:
            print(f"An unexpected error occurred loading awards: {e}")
    else:
        print(f"Warning: No awards file found matching 'sundt_awards_*.json' in '{directory}'")

    return {'projects': projects_data, 'awards': awards_data}

# Load the data from the specified data directory
results = load_data_from_latest_files(DATA_DIR)

# --- Embedding Model Initialization ---
# Initialize the embedding model using Google Gemini embeddings
# Set the Google API key from environment variable

# Use environment variable for API key
google_api_key = os.getenv('GOOGLE_API_KEY')

# Check if Google API key is available
if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    print("Please set your Google API key as an environment variable:")
    print("export GOOGLE_API_KEY='your-api-key-here'")
    embedding_model = None # Cannot proceed without API key
else:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # Specify the model explicitly
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
        print("Initialized GoogleGenerativeAIEmbeddings model: models/embedding-001")
    except ImportError:
        # Fallback to alternative Google embeddings if available (GooglePalmEmbeddings is older)
        try:
            from langchain.embeddings import GooglePalmEmbeddings
            embedding_model = GooglePalmEmbeddings(google_api_key=google_api_key)
            print("Initialized GooglePalmEmbeddings model")
        except ImportError:
            print("Error: Google embeddings not available. Please install langchain-google-genai:")
            print("pip install langchain-google-genai")
            embedding_model = None
    except Exception as e:
        print(f"Error initializing Google embeddings: {e}")
        embedding_model = None


# --- Document Preparation ---
# Prepare documents for vector store
documents = []

# Convert projects to documents using the correct structure from crawled data
if results and results.get('projects'):
    for project in results['projects']:
        # Create a comprehensive text representation of each project using actual fields
        project_text_parts = []

        # Add project name
        if project.get('project_name'):
            project_text_parts.append(f"Project Name: {project['project_name']}")

        # Add overview/description
        if project.get('overview'):
            project_text_parts.append(f"Overview: {project['overview']}")

        # Add metadata fields
        metadata_dict = project.get('metadata', {})
        for key, value in metadata_dict.items():
            if value:
                project_text_parts.append(f"{key}: {value}")

        # Add features
        features = project.get('features', [])
        if features:
            project_text_parts.append(f"Features: {'; '.join(features)}")

        # Add similar projects information
        similar_projects = project.get('similar_projects', [])
        if similar_projects:
            similar_titles = [sp.get('title', '') for sp in similar_projects if sp.get('title')]
            if similar_titles:
                project_text_parts.append(f"Related Projects: {'; '.join(similar_titles)}")

        project_text = '\n'.join(project_text_parts)

        # Create metadata for filtering and context using actual project structure
        metadata = {
            'type': 'project',
            'project_name': project.get('project_name', ''),
            'project_url': project.get('project_url', ''),
            'location': metadata_dict.get('Location', ''),
            'client': metadata_dict.get('Client', ''),
            'contractor': metadata_dict.get('Contractor', ''),
            'architect': metadata_dict.get('Architect', ''),
            'project_type': metadata_dict.get('Project Type', ''),
            'completion_date': metadata_dict.get('Completion Date', ''),
            'contract_value': metadata_dict.get('Contract Value', ''),
            'features_count': len(features),
            'similar_projects_count': len(similar_projects)
        }

        documents.append(Document(page_content=project_text.strip(), metadata=metadata))
else:
    print("No project data loaded.")


# Convert awards to documents using the correct structure from crawled data
if results and results.get('awards'):
    for award in results['awards']:
        # Create a comprehensive text representation of each award using actual fields
        award_text_parts = []

        # Add award title
        if award.get('title'):
            award_text_parts.append(f"Award Title: {award['title']}")

        # Add awarding organization
        if award.get('awarded_by'):
            award_text_parts.append(f"Awarded By: {award['awarded_by']}")

        # Add project name
        if award.get('project_name'):
            award_text_parts.append(f"Project Name: {award['project_name']}")

        # Add year
        if award.get('year'):
            award_text_parts.append(f"Year: {award['year']}")

        # Add category
        if award.get('category'):
            award_text_parts.append(f"Category: {award['category']}")

        # Add location
        if award.get('location'):
            award_text_parts.append(f"Location: {award['location']}")

        # Add raw details for additional context
        if award.get('raw_details'):
            award_text_parts.append(f"Details: {award['raw_details']}")

        award_text = '\n'.join(award_text_parts)

        # Create metadata for filtering and context using actual award structure
        metadata = {
            'type': 'award',
            'title': award.get('title', ''),
            'awarded_by': award.get('awarded_by', ''),
            'project_name': award.get('project_name', ''),
            'project_link': award.get('project_link', ''),
            'year': award.get('year', ''),
            'category': award.get('category', ''),
            'location': award.get('location', '')
        }

        documents.append(Document(page_content=award_text.strip(), metadata=metadata))
else:
    print("No award data loaded.")


print(f"Created {len(documents)} documents for vector store")
print(f"Projects loaded: {len(results.get('projects', []))}, Awards loaded: {len(results.get('awards', []))}")

# --- Vector Store Creation ---
# Create FAISS vector store only if we have a valid API key, embedding model, and documents
if google_api_key and embedding_model and documents:
    try:
        print("Creating FAISS vector store...")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        print("Vector store created successfully!")

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  # Return top 5 most similar documents
        )

        print("Retriever configured to return top 5 similar documents")

        # Save the vector store for later use
        print(f"Saving vector store locally to '{FAISS_INDEX_DIR}'...")
        vectorstore.save_local(FAISS_INDEX_DIR)
        print("Vector store saved successfully!")

    except Exception as e:
        print(f"Error creating or saving vector store: {e}")
        print("Please ensure your Google API key is valid and has sufficient credits.")
        vectorstore = None
        retriever = None
elif not google_api_key:
    print("Skipping vector store creation due to missing Google API key")
    vectorstore = None
    retriever = None
elif not embedding_model:
    print("Skipping vector store creation due to embedding model initialization failure")
    vectorstore = None
    retriever = None
else: # This case should only happen if documents list is empty
    print("No documents available to create vector store")
    vectorstore = None
    retriever = None

