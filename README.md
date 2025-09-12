# Daily Digest Agent

This project is a FastAPI application that generates a "daily digest" for a user, summarizing important information from the BuildMapper CRM.

## Deployed Site

You can access the deployed site here: [https://daily-digest-agent-ms5w.onrender.com/](https://daily-digest-agent-ms5w.onrender.com/)

## Features

*   **Daily Digest:** Provides a summary of:
    *   Upcoming meetings
    *   High-priority deals
    *   Approaching deadlines
*   **Recent Emails:** Shows recent emails related to CRM leads.
*   **Similar Leads:** Suggests new leads based on similarity to existing leads using OpenAI embeddings and ChromaDB.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Dailydigestagent
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    *   Copy the `.env.example` file to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and add your API keys and other configuration values.

## Usage

To run the application, use the following command:

```bash
uvicorn app:app --reload
```

The application will be available at `http://0.0.0.0:8000`.

## Project Structure

```
├── app.py                  # Main FastAPI application
├── daily_digest_agent.py   # Logic for generating the daily digest
├── lead_similarity.py      # Logic for finding similar leads
├── recent_emails.py        # Logic for fetching recent emails
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates for the web interface
│   ├── digest.html
│   ├── error.html
│   └── index.html
├── .env.example            # Example environment variables
└── README.md               # This file
```
