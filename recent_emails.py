import os
import sys
import json
import re
import html
import requests
from datetime import datetime, timedelta
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://crm.buildmapper.ai/api/v1/execute_query"
API_KEY = os.getenv("PROD_CRM_API_KEY")
HEADERS = {
    "API-Key": API_KEY,
    "Content-Type": "application/json",
}


def fetch_data(query: str, debug: bool = False) -> List[Dict]:
    payload = {"query": query}
    try:
        if debug:
            print("[DEBUG] Executing SQL:\n" + query)
        r = requests.post(API_URL, headers=HEADERS, json=payload)
        r.raise_for_status()
        data = r.json()
        rows = data.get("result", {}).get("data", []) or []
        if debug:
            print(f"[DEBUG] Returned {len(rows)} rows")
        return rows
    except requests.exceptions.RequestException:
        return []


def get_recent_emails_for_user(user_id: int, debug: bool = False) -> List[Dict]:
    """
    Fetch recent emails (last 72 hours) from mail_message where:
      - message_type = 'email'
      - model = 'crm.lead'
      - res_id joins to crm_lead.id
      - crm_lead.user_id = user_id
    Returns a list of dicts suitable for digest.html rendering.
    """
    # Basic guardrails
    if not API_KEY:
        # Missing API key will always return empty; surface a clear hint in CLI mode
        if __name__ == "__main__":
            print("[WARN] PROD_CRM_API_KEY is not set. Check your .env.")
        return []
    # Build query: latest email per lead (res_id) in last 72 hours for this user.
    # Simple single SELECT with NOT EXISTS per res_id (no CTEs / window functions).
    query = f"""
        SELECT 
            mm.id AS message_id,
            mm.subject,
            mm.body,
            mm.email_from,
            mm.date,
            mm.author_id,
            mm.message_type,
            mm.model,
            mm.res_id,
            cl.id AS lead_id,
            cl.name AS lead_name,
            cl.user_id AS lead_user_id,
            mm.create_uid
        FROM mail_message mm
        JOIN crm_lead cl ON mm.model = 'crm.lead' AND mm.res_id = cl.id
        WHERE mm.message_type = 'email'
          AND mm.model = 'crm.lead'
          AND mm.date >= NOW() - INTERVAL '72 hours'
          AND cl.user_id = {user_id}
          AND mm.create_uid != {user_id}
          -- ensure it's the latest message for that lead
          AND NOT EXISTS (
              SELECT 1
              FROM mail_message mm2
              WHERE mm2.model = 'crm.lead'
                AND mm2.message_type = 'comment'
                AND mm2.res_id = mm.res_id
                AND (
                    mm2.date > mm.date OR (mm2.date = mm.date AND mm2.id > mm.id)
            )
          )
        ORDER BY mm.date DESC
    """

    rows = fetch_data(query, debug=debug)
    emails: List[Dict] = []
    for row in rows:
        emails.append({
            "mail_id": row.get("message_id"),
            "subject": row.get("subject") or "No Subject",
            "body": row.get("body") or "",
            "email_from": row.get("email_from") or "",
            "date": row.get("date"),
            "author_id": row.get("author_id"),
            "message_type": row.get("message_type"),
            "model": row.get("model"),
            "record_name": row.get("lead_name"),
            "lead_id": row.get("lead_id"),
            "create_uid": row.get("create_uid"),
        })
    return emails

def _strip_html(html_text: str) -> str:
    if not html_text:
        return ""
    # Unescape HTML entities then remove tags
    unescaped = html.unescape(html_text)
    text = re.sub(r"<[^>]+>", " ", unescaped)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_user_name(user_id: int, debug: bool = False) -> str:
    """Fetches the name of a user from res_users."""
    query = f"SELECT name FROM res_users WHERE id = {user_id} LIMIT 1"
    rows = fetch_data(query, debug=debug)
    return rows[0].get("name") if rows else "Unknown User"


def get_author_name(author_id: int, debug: bool = False) -> str:
    """Fetches the name of an author from res_partner."""
    query = f"SELECT name FROM res_partner WHERE id = {author_id} LIMIT 1"
    rows = fetch_data(query, debug=debug)
    return rows[0].get("name") if rows else "Unknown Author"


def clean_email_body(body: str, max_len: int = 300) -> str:
    """
    Cleans the email body by removing quoted replies and truncating.
    """
    if not body:
        return ""
    
    # Strip HTML tags first
    text = _strip_html(body)
    
    # Remove quoted replies (e.g., "On ... wrote:")
    text = re.split(r"On .* wrote:", text, flags=re.IGNORECASE)[0]
    
    # Additional common reply patterns
    text = re.split(r"From:.*", text, flags=re.IGNORECASE)[0]
    text = re.split(r"Sent from my.*", text, flags=re.IGNORECASE)[0]
    
    # Remove long signature-like separators
    text = re.sub(r"_{10,}", "", text)
    
    # Normalize whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    
    # Truncate if necessary
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def get_recent_emails_for_user_clean(user_id: int, body_preview_chars: int = 300, debug: bool = False) -> List[Dict]:
    """
    Convenience wrapper that also strips HTML, cleans the body, and enriches with author/user names.
    """
    emails = get_recent_emails_for_user(user_id, debug=debug)
    
    # Get user's name once
    user_name = get_user_name(user_id, debug=debug)
    
    cleaned: List[Dict] = []
    for e in emails:
        # Clean the email body
        body_cleaned = clean_email_body(e.get("body", ""), max_len=body_preview_chars)
        
        # Get author's name
        author_name = get_author_name(e.get("author_id"), debug=debug)
        
        cleaned.append({
            **e,
            "body": body_cleaned,
            "user_name": user_name,
            "author_name": author_name,
        })
        
    return cleaned


def main():
    if len(sys.argv) < 2:
        print("Usage: python recent_emails.py <user_id> [--clean] [--hours=N] [--debug]")
        sys.exit(1)
    try:
        user_id = int(sys.argv[1])
    except ValueError:
        print("user_id must be an integer")
        sys.exit(1)

    # flags
    use_clean = False
    debug_flag = False

    for arg in sys.argv[2:]:
        if arg == "--clean":
            use_clean = True
        elif arg == "--debug":
            debug_flag = True

    if use_clean:
        result = get_recent_emails_for_user_clean(user_id, debug=debug_flag)
    else:
        result = get_recent_emails_for_user(user_id, debug=debug_flag)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
