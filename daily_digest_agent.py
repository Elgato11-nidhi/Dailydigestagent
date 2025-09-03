import json
import requests
from datetime import datetime, timedelta
import sys 
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_data(query):
    """Fetches data from the API using a SQL query."""
    url = "https://staging.crm.buildmapper.ai/api/v1/execute_query"
    headers = {
        "API-Key": os.getenv('CRM_API_KEY'),
        "Content-Type": "application/json"
    }
    payload = {"query": query}
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get('result', {}).get('data', [])
        else:
            error_message = f"Error: Unable to fetch data. Status code: {response.status_code}\nResponse: {response.text}"
            return [{"error": error_message}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"An error occurred during the request: {e}"}]


def get_upcoming_meetings(calendar_events, hours=72):
    now = datetime.now()
    upcoming = []
    for event in calendar_events:
        start_str = event.get("start")
        if not start_str:
            continue
        try:
            start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if now <= start <= now + timedelta(hours=hours):
            upcoming.append(event)
    return upcoming


def get_high_priority_deals(crm_leads, top_n=5):
    # Separate leads by priority: priority 1+ first, then priority 0
    priority_leads = []
    zero_priority_leads = []
    
    for lead in crm_leads:
        priority = lead.get("priority")
        if priority is not None:
            if int(priority) >= 1:
                priority_leads.append(lead)
            elif int(priority) == 0:
                zero_priority_leads.append(lead)
    
    # Sort priority leads by priority (highest first)
    priority_leads.sort(key=lambda x: int(x.get("priority", 0)), reverse=True)
    
    # Sort zero priority leads by name (alphabetical)
    zero_priority_leads.sort(key=lambda x: x.get("name", "").lower())
    
    # Combine: priority 1+ first, then priority 0
    all_leads = priority_leads + zero_priority_leads
    
    return all_leads[:top_n]


def get_deadlines(crm_leads, days=2):
    now = datetime.now().date()
    soon = []
    for lead in crm_leads:
        deadline_str = lead.get("date_deadline")
        if not deadline_str or not isinstance(deadline_str, str):
            continue
        try:
            deadline = datetime.strptime(deadline_str.split(" ")[0], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            continue
        if now <= deadline <= now + timedelta(days=days):
            soon.append(lead)
    return soon


def partner_name(partner_ids, partners):
    """Get partner name(s) from partner IDs (can be a single ID or list)."""
    if not partner_ids:
        return "null"
    
    if isinstance(partner_ids, list):
        names = []
        for pid in partner_ids:
            if pid is not None:
                for p in partners:
                    if p.get("companyId") == pid:
                        name = p.get("companyName") or "null"
                        if name != "null":
                            names.append(name)
                        break
        return ", ".join(names) if names else "null"
    else:
        for p in partners:
            if p.get("companyId") == partner_ids:
                return p.get("companyName") or "null"
        return "null"


def safe_get(d, key):
    v = d.get(key)
    return v if v is not None else "null"

def get_digest(user_id):
    """
    Generates the daily digest for a given user ID.
    Returns the digest as a dictionary of lists.
    """
    digest_data = {
        "upcoming_meetings": [],
        "high_priority_deals": [],
        "deadlines_soon": [],
        "errors": []
    }

    calendar_events_data = fetch_data(f"SELECT * FROM calendar_event WHERE user_id = {user_id}")
    if calendar_events_data and len(calendar_events_data) > 0 and "error" in calendar_events_data[0]:
        digest_data["errors"].append(calendar_events_data[0]["error"])
        return digest_data
    calendar_events = calendar_events_data

    crm_leads_data = fetch_data(
        f"SELECT id, name, priority, builder_company, builder_partner_id, date_deadline, expected_revenue FROM crm_lead WHERE user_id = {user_id} AND active = 'true'"
    )
    if crm_leads_data and len(crm_leads_data) > 0 and "error" in crm_leads_data[0]:
        digest_data["errors"].append(crm_leads_data[0]["error"])
        return digest_data
    crm_leads = crm_leads_data

    partner_ids = set()
    for lead in crm_leads:
        if lead.get("builder_partner_id"):
            if isinstance(lead.get("builder_partner_id"), list):
                partner_ids.update(p_id for p_id in lead.get("builder_partner_id") if p_id is not None)
            else:
                partner_ids.add(lead.get("builder_partner_id"))

    if partner_ids:
        partner_ids_str = ",".join(str(pid) for pid in partner_ids)
        partners_data = fetch_data(
            f"SELECT companyId, companyName FROM res_partner WHERE id IN ({partner_ids_str})"
        )
        if partners_data and len(partners_data) > 0 and "error" in partners_data[0]:
            digest_data["errors"].append(partners_data[0]["error"])
            partners = []
        else:
            partners = partners_data
    else:
        partners = []

    upcoming = get_upcoming_meetings(calendar_events)
    for event in upcoming:
        name = safe_get(event, "name")
        start = safe_get(event, "start")
        opportunity_id = event.get("opportunity_id")
        deal_name = ""
        if opportunity_id:
            deal = next((lead for lead in crm_leads if lead.get("id") == opportunity_id), None)
            if deal:
                deal_name = safe_get(deal, "name")
        
        company_id = event.get("company_id")
        builder_company = ""
        if company_id:
            lead_data = fetch_data(f"SELECT builder_company FROM crm_lead WHERE company_id = {company_id}")
            if lead_data and len(lead_data) > 0 and "error" not in lead_data[0] and lead_data[0].get("builder_company"):
                builder_company = lead_data[0].get("builder_company")
        
        digest_data["upcoming_meetings"].append({
            "name": name,
            "deal_name": deal_name,
            "builder_company": builder_company,
            "start": start
        })

    high_priority = get_high_priority_deals(crm_leads)
    for lead in high_priority:
        pname = safe_get(lead, 'builder_company')
        if pname == "null" and lead.get("builder_partner_id"):
            pname = partner_name(lead.get("builder_partner_id"), partners)
        lead_name = safe_get(lead, 'name')
        lead_revenue = safe_get(lead, 'expected_revenue')
        lead_priority = lead.get("priority", 0)
        digest_data["high_priority_deals"].append({
            "name": lead_name,
            "revenue": lead_revenue,
            "partner_name": pname,
            "priority": lead_priority
        })

    deadlines = get_deadlines(crm_leads)
    for lead in deadlines:
        pname = safe_get(lead, 'builder_company')
        if pname == "null" and lead.get("builder_partner_id"):
            pname = partner_name(lead.get("builder_partner_id"), partners)
        lead_name = safe_get(lead, 'name')
        deadline = safe_get(lead, 'date_deadline')
        lead_priority = lead.get("priority", 0)
        digest_data["deadlines_soon"].append({
            "name": lead_name,
            "deadline": deadline,
            "partner_name": pname,
            "priority": lead_priority
        })

    return digest_data

def main():
    if len(sys.argv) < 2:
        print("Usage: python daily_digest_agent.py <user_id>")
        sys.exit(1)
    try:
        user_id = int(sys.argv[1])
    except ValueError:
        print("User ID must be an integer.")
        sys.exit(1)

    digest_output = get_digest(user_id)
    print(json.dumps(digest_output, indent=4))


if __name__ == "__main__":
    main()
