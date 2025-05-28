import os
import json
from typing import Optional, Dict, Any, List
import re
import logging
import time
from datetime import datetime
from collections import defaultdict, deque

# Import necessary libraries for the backend application
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain components for RAG
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# Fix deprecated imports for FAISS, PromptTemplate and Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document # For type hinting

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Directory where the FAISS index is saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_DIR = os.path.join(SCRIPT_DIR, "faiss_index")

# Get Google API key from environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(SCRIPT_DIR, 'rag_system.log')), # Log to a file in the script's directory
        logging.StreamHandler()
    ]
)
rag_logger = logging.getLogger('RAG_System_Backend')

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Sundt RAG Backend",
    description="Backend API for querying Sundt project and award data using RAG with specialized agents and metrics.",
    version="1.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- RAG Metrics ---
class RAGMetrics:
    """Metrics collection and monitoring for RAG system"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.query_history = deque(maxlen=max_history)
        self.sanitization_applied_count = 0 # Count of queries where sanitization changed the input
        self.response_times = deque(maxlen=max_history)
        self.query_counts = defaultdict(int)
        
    def log_query(self, original_query: str, processed_query: str, agent_type: str, response_time: float, 
                  sanitization_applied: bool = False, error: Optional[str] = None):
        """Log query metrics and details"""
        timestamp = datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "original_query": original_query,
            "processed_query": processed_query,
            "agent_type": agent_type,
            "response_time": response_time,
            "sanitization_applied": sanitization_applied,
            "error": error
        }
        
        self.query_history.append(log_entry)
        self.response_times.append(response_time)
        self.query_counts[agent_type] += 1
        
        if sanitization_applied:
            self.sanitization_applied_count +=1
            rag_logger.warning(f"Sanitization applied to query. Original: {original_query[:100]}..., Processed: {processed_query[:100]}...")
        
        if error:
            rag_logger.error(f"Query failed - Agent: {agent_type}, Time: {response_time:.2f}s, Error: {error}, Original Query: {original_query[:100]}...")
        else:
            rag_logger.info(f"Query processed - Agent: {agent_type}, Time: {response_time:.2f}s, Original Query: {original_query[:100]}...")

# --- Global Variables for RAG Components ---
embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None
llm: Optional[ChatGoogleGenerativeAI] = None
vectorstore: Optional[FAISS] = None
projects_agent: Optional[RetrievalQA] = None
awards_agent: Optional[RetrievalQA] = None
rag_metrics: Optional[RAGMetrics] = None


# --- Input Sanitization and Validation ---
def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input to prevent prompt injection attacks while preserving legitimate queries."""
    if not isinstance(user_input, str):
        return ""
    
    sanitized = user_input.strip()
    
    # More targeted jailbreak patterns - less aggressive
    jailbreak_patterns = [
        r'ignore\s+(all\s+)?previous\s+instructions?',
        r'forget\s+(all\s+)?instructions?',
        r'system\s*:\s*ignore',
        r'assistant\s*:\s*ignore',
        r'new\s+instructions?\s*:',
        r'override\s+instructions?',
        r'disregard\s+(all\s+)?previous',
    ]
    
    for pattern in jailbreak_patterns:
        sanitized = re.sub(pattern, '[SANITIZED]', sanitized, flags=re.IGNORECASE)
    
    # Remove only dangerous characters, keep more punctuation for natural queries
    sanitized = re.sub(r'[<>{}|\\^`~]', '', sanitized)
    
    # Increased max length for more detailed queries
    max_length = 800
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized.strip()

def validate_query_type(query: str, expected_domain: str) -> bool:
    """Validate that the query is appropriate for the expected domain."""
    if not query or len(query.strip()) < 1:
        return False
    
    # Check for completely empty or meaningless queries
    if query.strip().lower() in ['hi', 'hello', 'hey', 'test', '']:
        return False
    
    # Allow most queries through - let the LLM handle domain filtering
    return True

# Removed convert_markdown_to_html function as per instructions.
# The LLM will now be instructed to output plain text.

def detect_jailbreak_attempt(query: str) -> bool:
    """Detect potential jailbreak attempts."""
    jailbreak_indicators = [
        r'ignore\s+(all\s+)?previous',
        r'forget\s+(all\s+)?instructions',
        r'system\s*:',
        r'assistant\s*:',
        r'new\s+instructions?\s*:',
        r'override\s+instructions',
        r'disregard\s+(all\s+)?previous',
        r'confidential\s+information',
        r'tell\s+me\s+about\s+.*secret',
    ]
    
    for pattern in jailbreak_indicators:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False

# --- Prompt Templates ---
# Modified projects prompt: Removed instruction for LLM to generate HTML
projects_prompt_template_text = """You are a specialized assistant that answers questions about Sundt Construction's past projects based solely on the provided context.

INSTRUCTIONS:
- Answer questions about construction projects, timelines, clients, project outcomes, sustainability initiatives, and project details.
- Base your response exclusively on the context provided below.
- Format your response as plain text. You may use standard markdown formatting like bullet points (*) or bolding (**text**) if helpful, but do NOT use HTML tags.
- Do NOT include URLs or links directly in your answer text. The system will handle linking based on source documents.
- If the context doesn't contain relevant information, respond with: "I don't have information about that specific project in my database."
- For greetings or off-topic queries, respond with: "I specialize in providing information about Sundt's construction projects. How can I help you with project-related questions?"

Context from project database:
{context}

User question about projects: {question}

Response:"""
PROJECTS_PROMPT = PromptTemplate(input_variables=["context", "question"], template=projects_prompt_template_text)

# Modified awards prompt: Removed instruction for LLM to generate HTML
awards_prompt_template_text = """You are a specialized assistant that answers questions about Sundt Construction's awards and recognitions based solely on the provided context.

INSTRUCTIONS:
- Answer questions about awards, recognitions, honors, achievements, and accolades.
- Base your response exclusively on the context provided below.
- Format your response as plain text. You may use standard markdown formatting like bullet points (*) or bolding (**text**) if helpful, but do NOT use HTML tags.
- If the context doesn't contain relevant information, respond with: "I don't have information about that specific award in my database."
- For greetings or off-topic queries, respond with: "I specialize in providing information about Sundt's awards and recognitions. How can I help you with award-related questions?"

Context from awards database:
{context}

User question about awards: {question}

Response:"""
AWARDS_PROMPT = PromptTemplate(input_variables=["context", "question"], template=awards_prompt_template_text)

# --- Helper Functions for Retrievers ---
def detect_comprehensive_query(query: str) -> bool:
    """Detect if user is asking for comprehensive information that requires more documents."""
    comprehensive_indicators = [
        r'\ball\s+projects?\b',
        r'\bevery\s+project\b',
        r'\bcomplete\s+list\b',
        r'\bfull\s+list\b',
        r'\btotal\s+projects?\b',
        r'\bhow\s+many\s+projects?\b',
        r'\blist\s+(all\s+)?projects?\b',
        r'\bshow\s+me\s+(all\s+)?projects?\b',
        r'\btell\s+me\s+about\s+(all\s+)?projects?\b',
        r'\bwhat\s+projects?\b',
        r'\boverview\s+of\s+projects?\b',
        r'\bsummary\s+of\s+projects?\b',
    ]
    
    for pattern in comprehensive_indicators:
        if re.search(pattern, query, re.IGNORECASE):
            return True
    return False

def create_projects_retriever(vs: FAISS, k: int = 5):
    """Create a projects retriever with configurable k value."""
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k, "filter": {"type": "project"}})

def create_awards_retriever(vs: FAISS, k: int = 5):
    """Create an awards retriever with configurable k value."""
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k, "filter": {"type": "award"}})

def get_optimal_k_value(query: str, agent_type: str) -> int:
    """Determine optimal number of documents to retrieve based on query type."""
    if detect_comprehensive_query(query):
        # For comprehensive queries, retrieve more documents
        if agent_type == "projects":
            return 15  # Increased for comprehensive project overviews
        else:
            return 10  # For awards
    else:
        # For specific queries, use fewer documents for better relevance
        return 5

# --- Startup Event: Load Models and Vector Store ---
@app.on_event("startup")
async def startup_event():
    global embedding_model, llm, vectorstore, projects_agent, awards_agent, rag_metrics

    rag_logger.info("Application startup: Loading RAG components...")
    rag_metrics = RAGMetrics()

    if not GOOGLE_API_KEY:
        rag_logger.error("Error: GOOGLE_API_KEY environment variable not found.")
        return

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")
        rag_logger.info("Initialized GoogleGenerativeAIEmbeddings model.")

        if not os.path.exists(FAISS_INDEX_DIR):
            rag_logger.error(f"Error: FAISS index directory not found at {FAISS_INDEX_DIR}. Run create_vector_store.py.")
            return
        vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
        rag_logger.info("Loaded FAISS vector store.")

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1, convert_system_message_to_human=True)
        rag_logger.info("Initialized ChatGoogleGenerativeAI model (Gemini Pro).")

        # Create specialized retrievers
        # Use get_optimal_k_value here if you want startup to use a specific k,
        # but it's usually better to determine k per query in process_query.
        # For startup, we just need the agents initialized with a default retriever.
        # The k value will be dynamically set in process_query by creating a new retriever instance.
        projects_retriever_default = create_projects_retriever(vectorstore, k=5) # Default k for initialization
        awards_retriever_default = create_awards_retriever(vectorstore, k=5) # Default k for initialization
        rag_logger.info("Created default specialized retrievers for projects and awards.")

        # Past Projects Agent
        projects_agent = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=projects_retriever_default, # Use default retriever for init
            chain_type_kwargs={"prompt": PROJECTS_PROMPT}, return_source_documents=True
        )
        rag_logger.info("Initialized Projects RAG Agent.")

        # Awards Agent
        awards_agent = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=awards_retriever_default, # Use default retriever for init
            chain_type_kwargs={"prompt": AWARDS_PROMPT}, return_source_documents=True
        )
        rag_logger.info("Initialized Awards RAG Agent.")
        rag_logger.info("All RAG components loaded successfully.")

    except Exception as e:
        rag_logger.error(f"Error during startup loading: {e}", exc_info=True)
        embedding_model, llm, vectorstore, projects_agent, awards_agent = None, None, None, None, None
        rag_logger.error("RAG components failed to load. API will not be fully functional.")


# --- API Endpoint Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: Optional[List[Dict[str, Any]]] = None
    sanitized_query: Optional[str] = None
    error: Optional[str] = None


# --- API Endpoints ---
async def process_query(
    original_query: str, 
    agent: Optional[RetrievalQA], 
    agent_type: str,
    expected_domain: str,
    domain_specific_error_message: str
) -> QueryResponse:
    
    if agent is None or rag_metrics is None or vectorstore is None: # Added vectorstore check
        rag_logger.error(f"{agent_type.capitalize()} agent, RAGMetrics, or vectorstore not available.")
        raise HTTPException(status_code=503, detail=f"Backend is not ready. {agent_type.capitalize()} RAG components failed to load.")

    start_time = time.time()
    
    # Check for jailbreak attempts first
    if detect_jailbreak_attempt(original_query):
        response_time = time.time() - start_time
        dry_response = "I can only provide information about Sundt Construction data."
        rag_metrics.log_query(
            original_query=original_query, processed_query="[JAILBREAK_DETECTED]", agent_type=agent_type,
            response_time=response_time, sanitization_applied=True, error="Jailbreak attempt detected"
        )
        return QueryResponse(answer=dry_response, source_documents=[], sanitized_query="[JAILBREAK_DETECTED]", error="Invalid query")
    
    sanitized_query = sanitize_user_input(original_query)
    sanitization_applied = original_query != sanitized_query
    error_msg_response = None

    try:
        if not validate_query_type(sanitized_query, expected_domain):
            # Handle greetings more gracefully
            if sanitized_query.strip().lower() in ['hi', 'hello', 'hey']:
                friendly_response = f"Hello! I specialize in providing information about Sundt's {agent_type}. How can I help you today?"
            else:
                friendly_response = domain_specific_error_message
                
            response_time = time.time() - start_time
            rag_metrics.log_query(
                original_query=original_query, processed_query=sanitized_query, agent_type=agent_type,
                response_time=response_time, sanitization_applied=sanitization_applied, error="Validation failed"
            )
            return QueryResponse(answer=friendly_response, source_documents=[], sanitized_query=sanitized_query, error=None)

        # Check if this is a comprehensive query that would benefit from special handling
        if detect_comprehensive_query(sanitized_query) and agent_type == "projects":
            # For comprehensive project queries, try to get a summary first
            try:
                comprehensive_summary = await get_all_projects_summary()
                response_time = time.time() - start_time
                rag_metrics.log_query(
                    original_query=original_query, processed_query=sanitized_query, agent_type=agent_type,
                    response_time=response_time, sanitization_applied=sanitization_applied
                )
                return QueryResponse(answer=comprehensive_summary, source_documents=[], sanitized_query=sanitized_query)
            except Exception as e:
                rag_logger.warning(f"Failed to get comprehensive summary, falling back to normal retrieval: {e}")
                # Fall through to normal processing

        # Determine optimal k value for this specific query
        optimal_k = get_optimal_k_value(sanitized_query, agent_type)
        
        # Create a retriever with the optimal k value for this query
        if agent_type == "projects":
            current_retriever = create_projects_retriever(vectorstore, k=optimal_k)
        elif agent_type == "awards":
            current_retriever = create_awards_retriever(vectorstore, k=optimal_k)
        else:
             # Should not happen with current setup, but good practice
             current_retriever = agent.retriever # Fallback to agent's default retriever

        # Temporarily update the agent's retriever for this query
        original_retriever = agent.retriever
        agent.retriever = current_retriever

        try:
            result = agent.invoke({"query": sanitized_query})
            raw_answer = result.get("result", f"Could not generate an answer for {agent_type}.")
            
            # Removed markdown to HTML conversion as per instructions.
            # The raw_answer from the LLM (instructed to use plain text/markdown) is returned directly.
            formatted_answer = raw_answer 
            
            source_docs_raw = result.get("source_documents", [])
            
            formatted_sources = []
            if isinstance(source_docs_raw, list):
                for doc in source_docs_raw:
                    if isinstance(doc, Document): # Ensure it's a Langchain Document
                        source_dict = {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        # If it's a project query and the source has a URL, add it explicitly
                        if agent_type == "projects" and "url" in doc.metadata:
                             source_dict["url"] = doc.metadata["url"]
                        formatted_sources.append(source_dict)
                    elif isinstance(doc, dict): # If already a dict (less likely from chain)
                         # Assuming metadata is already included in the dict
                         formatted_sources.append(doc)
                         # If it's a project query and the source has a URL, add it explicitly
                         if agent_type == "projects" and "url" in doc.get("metadata", {}):
                              doc["url"] = doc["metadata"]["url"]

        finally:
            # Restore the original retriever on the agent
            agent.retriever = original_retriever


        response_time = time.time() - start_time
        rag_metrics.log_query(
            original_query=original_query, processed_query=sanitized_query, agent_type=agent_type,
            response_time=response_time, sanitization_applied=sanitization_applied
        )
        # The frontend will need to be updated to read the 'url' key from the source_documents list
        # and display it as a link alongside the source content if present.
        # The 'answer' field now contains plain text/markdown, not HTML.
        return QueryResponse(answer=formatted_answer, source_documents=formatted_sources, sanitized_query=sanitized_query)

    except Exception as e:
        response_time = time.time() - start_time
        error_detail = f"An error occurred while processing your {agent_type} query. Please try again later."
        rag_logger.error(f"Error processing {agent_type} query: {e}. Original query: {original_query}", exc_info=True)
        rag_metrics.log_query(
            original_query=original_query, processed_query=sanitized_query, agent_type=agent_type,
            response_time=response_time, sanitization_applied=sanitization_applied, error=str(e)
        )
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/api/v1/rag/ask/projects", response_model=QueryResponse)
async def ask_projects_question(request: QueryRequest):
    """Handles queries about Sundt's past projects."""
    return await process_query(
        original_query=request.query,
        agent=projects_agent,
        agent_type="projects",
        expected_domain="projects",
        domain_specific_error_message="I specialize in providing information about Sundt's construction projects. How can I help you with project-related questions?"
    )

@app.post("/api/v1/rag/ask/awards", response_model=QueryResponse)
async def ask_awards_question(request: QueryRequest):
    """Handles queries about Sundt's awards and recognitions."""
    return await process_query(
        original_query=request.query,
        agent=awards_agent,
        agent_type="awards",
        expected_domain="awards",
        domain_specific_error_message="I specialize in providing information about Sundt's awards and recognitions. How can I help you with award-related questions?"
    )

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "Sundt RAG Backend is running. Use /api/v1/rag/ask/projects or /api/v1/rag/ask/awards to query."}

@app.get("/health")
async def health_check():
    """Health check endpoint to verify all components are loaded."""
    global embedding_model, llm, vectorstore, projects_agent, awards_agent, rag_metrics
    
    status = {
        "status": "healthy",
        "components": {
            "embedding_model": embedding_model is not None,
            "llm": llm is not None,
            "vectorstore": vectorstore is not None,
            "projects_agent": projects_agent is not None,
            "awards_agent": awards_agent is not None,
            "rag_metrics": rag_metrics is not None
        },
        "google_api_key_configured": GOOGLE_API_KEY is not None
    }
    
    # Check if all components are loaded
    all_loaded = all(status["components"].values()) and status["google_api_key_configured"]
    
    if not all_loaded:
        status["status"] = "unhealthy"
        return JSONResponse(status_code=503, content=status)
    
    return status

@app.get("/metrics")
async def get_metrics():
    """Get RAG system metrics."""
    global rag_metrics
    
    if rag_metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    # Calculate average response time
    avg_response_time = sum(rag_metrics.response_times) / len(rag_metrics.response_times) if rag_metrics.response_times else 0
    
    return {
        "total_queries": len(rag_metrics.query_history),
        "sanitization_applied_count": rag_metrics.sanitization_applied_count,
        "average_response_time": round(avg_response_time, 3),
        "query_counts_by_agent": dict(rag_metrics.query_counts),
        "recent_queries": [
            {
                "timestamp": entry["timestamp"],
                "agent_type": entry["agent_type"],
                "response_time": entry["response_time"],
                "sanitization_applied": entry["sanitization_applied"],
                "error": entry["error"]
            }
            for entry in list(rag_metrics.query_history)[-10:]  # Last 10 queries
        ]
    }

# --- Enhanced Query Processing ---
async def get_all_projects_summary() -> str:
    """Get a summary of all projects when user asks for comprehensive information."""
    global vectorstore
    
    if vectorstore is None:
        return "Vector store not available."
    
    try:
        # Try to get all project documents (or a large number)
        # Use a higher k value for comprehensive summary
        all_projects_retriever = create_projects_retriever(vectorstore, k=30)  # Increased to get more projects
        
        # Use a broad query to get diverse projects
        broad_query = "construction projects timeline client outcomes"
        docs = all_projects_retriever.get_relevant_documents(broad_query)
        
        if not docs:
            return "No project information found in the database."
        
        # Create a summary of projects
        project_names = set()
        for doc in docs:
            metadata = doc.metadata
            if 'project_name' in metadata:
                project_names.add(metadata['project_name'])
            elif 'title' in metadata:
                project_names.add(metadata['title'])
        
        if project_names:
            projects_list = sorted(list(project_names))
            # Format as plain text list
            summary = f"Found {len(projects_list)} projects in the database:\n\n"
            for project in projects_list:
                summary += f"- {project}\n"
            summary += "\nAsk me about any specific project for more details!"
            return summary
        else:
            return f"Found {len(docs)} project documents. Ask me about specific aspects like timelines, clients, or outcomes for detailed information."
            
    except Exception as e:
        rag_logger.error(f"Error getting all projects summary: {e}")
        return "Unable to retrieve comprehensive project information at this time."