from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
import os
import asyncio
import time

load_dotenv()

app = FastAPI(title="Daily Digest Agent", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Daily Digest Agent is running"}

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return {
        "service": "Daily Digest Agent",
        "version": "1.0.0",
        "status": "operational",
        "deployment": "vercel"
    }

@app.post("/api/digest")
async def get_digest_endpoint(user_id: int = Form(...)):
    """Get digest data for a user"""
    try:
        # Placeholder for actual digest logic
        # In production, this would call your ML models
        return {
            "user_id": user_id,
            "digest_data": {
                "message": "Digest functionality requires full deployment with ML dependencies",
                "status": "placeholder"
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/leads")
async def get_leads_endpoint(user_id: int = Form(...)):
    """Get leads data for a user"""
    try:
        # Placeholder for actual leads logic
        return {
            "user_id": user_id,
            "leads_data": {
                "message": "Leads functionality requires full deployment with ML dependencies",
                "status": "placeholder"
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/emails")
async def get_emails_endpoint(user_id: int = Form(...)):
    """Get emails data for a user"""
    try:
        # Placeholder for actual emails logic
        return {
            "user_id": user_id,
            "emails_data": {
                "message": "Emails functionality requires full deployment with ML dependencies",
                "status": "placeholder"
            }
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
