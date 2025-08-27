from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
import os
import sys
import asyncio
import concurrent.futures
from functools import partial
import time

load_dotenv()

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Import after env vars are loaded
from daily_digest_agent import get_digest
from lead_similarity import LeadSimilarityAnalyzer
from activity import EmailActivityFetcher


def run_in_executor(func, *args, **kwargs):
    """Helper function to run synchronous functions in executor"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


async def get_all_data_parallel(user_id: int):
    """Fetch all data in parallel using ThreadPoolExecutor for better performance"""
    start_time = time.time()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Create tasks for parallel execution
            digest_task = run_in_executor(get_digest, user_id)
            leads_task = run_in_executor(LeadSimilarityAnalyzer().find_interest_matched_new_leads, user_id, 0.7)
            emails_task = run_in_executor(EmailActivityFetcher().get_emails_clean, user_id)
            
            # Execute all tasks concurrently and wait for all to complete
            results = await asyncio.gather(
                digest_task, leads_task, emails_task,
                return_exceptions=True
            )
            
            # Handle any exceptions that occurred during execution
            digest_output, similar_matches, emails_data = results
            
            # Check for exceptions and provide fallbacks
            if isinstance(digest_output, Exception):
                print(f"Error in digest: {digest_output}")
                digest_output = {"error": "Failed to fetch digest data"}
            
            if isinstance(similar_matches, Exception):
                print(f"Error in leads: {similar_matches}")
                similar_matches = []
            
            if isinstance(emails_data, Exception):
                print(f"Error in emails: {emails_data}")
                emails_data = []
            
            end_time = time.time()
            print(f"Parallel execution completed in {end_time - start_time:.2f} seconds")
            
            return {
                "digest_data": digest_output,
                "similar_leads_data": similar_matches,
                "emails_data": emails_data,
                "execution_time": end_time - start_time
            }
            
    except Exception as e:
        print(f"Error in parallel execution: {e}")
        # Fallback to sequential execution if parallel fails
        try:
            print("Falling back to sequential execution...")
            digest_output = get_digest(user_id)
            similar_matches = LeadSimilarityAnalyzer().find_interest_matched_new_leads(user_id, 0.7)
            emails_data = EmailActivityFetcher().get_emails_clean(user_id)
            
            return {
                "digest_data": digest_output,
                "similar_leads_data": similar_matches,
                "emails_data": emails_data,
                "execution_time": "sequential_fallback"
            }
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return {
                "digest_data": {"error": "Failed to fetch data"},
                "similar_leads_data": [],
                "emails_data": [],
                "execution_time": "error"
            }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def submit(request: Request, user_id: int = Form(...)):
    try:
        # Get all data in parallel
        all_data = await get_all_data_parallel(user_id)
        
        return templates.TemplateResponse("digest.html", {
            "request": request,
            "user_id": user_id,
            "digest_data": all_data["digest_data"],
            "similar_leads_data": all_data["similar_leads_data"],
            "emails_data": all_data["emails_data"],
            "execution_time": all_data.get("execution_time", "unknown")
        })
    except Exception as e:
        import traceback
        exc_type, exc_value, exc_traceback = sys.exc_info()
        # Get last traceback frame for accurate error location
        tb_last = traceback.extract_tb(exc_traceback)[-1]
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": f"Error: {e}",
            "line_number": tb_last.lineno,
            "file_name": tb_last.filename
        })


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/test-chromadb-v2")
async def test_chromadb_v2():
    """Test ChromaDB v2 connection"""
    try:
        from lead_similarity import LeadSimilarityAnalyzer
        
        analyzer = LeadSimilarityAnalyzer()
        connection_status = analyzer.test_connection()
        
        return {
            "status": "success",
            "chromadb_status": connection_status,
            "message": "ChromaDB v2 connection test completed",
            "api_endpoint": "api.trychroma.com:443"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to test ChromaDB v2: {str(e)}"
        }


@app.get("/test-openai")
async def test_openai():
    """Test OpenAI client initialization"""
    try:
        from openai import OpenAI
        
        # Test basic OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        return {
            "status": "success",
            "message": "OpenAI client initialized successfully",
            "api_key_length": len(os.getenv('OPENAI_API_KEY', '')),
            "embedding_models": [
                "text-embedding-3-small",
                "text-embedding-3-large", 
                "text-embedding-ada-002"
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAI client initialization failed: {str(e)}",
            "error_type": type(e).__name__
        }


if __name__ == "__main__":
    # For production deployment on Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)