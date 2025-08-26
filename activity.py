import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import re
import asyncio
import aiohttp

load_dotenv()

class EmailActivityFetcher:
    def __init__(self):
        self.api_url = "https://staging.crm.buildmapper.ai/api/v1/execute_query"
        self.headers = {
            "API-Key": os.getenv('CRM_API_KEY'),
            "Content-Type": "application/json"
        }
    
    def extract_email_from_string(self, email_string):
        """Extract email address from formatted string like 'Name <email@domain.com>'"""
        if not email_string:
            return None
        
        # Try to extract email from format "Name <email@domain.com>"
        email_match = re.search(r'<([^>]+@[^>]+)>', email_string)
        if email_match:
            return email_match.group(1)
        
        # If no angle brackets, check if it's just an email
        if '@' in email_string and re.match(r'^[^<>]+@[^<>]+$', email_string.strip()):
            return email_string.strip()
        
        return None
    
    def get_user_emails(self, user_id):
        """Get email addresses for a specific user from res_users table"""
        query = f"""
        SELECT id, login, partner_id, company_id, active
        FROM res_users
        WHERE id = {user_id} AND active = true
        """
        
        payload = {"query": query}
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                if data.get("result", {}).get("success") and data["result"]["data"]:
                    user_data = data["result"]["data"][0]
                    return user_data.get("login")  # This should be the user's email
                else:
                    print(f"No user found with ID {user_id}")
                    return None
            else:
                print(f"Error fetching user data: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting user emails: {e}")
            return None
    
    def get_recent_emails(self, user_email):
        """Get recent emails for a specific user from mail_mail and mail_message tables"""
        if not user_email:
            print("No user email provided")
            return []
        
        print(f"Looking for emails sent to: {user_email}")
        
        # First, get emails from mail_mail table where email_to matches the user's email
        # The email_to field contains formatted strings like "Name <email@domain.com>"
        mail_query = f"""
        SELECT id, mail_message_id, email_to, write_date FROM mail_mail WHERE write_date >= NOW() - INTERVAL '2 days'
        """
        
        try:
            # Get emails from mail_mail table
            mail_payload = {"query": mail_query}
            mail_response = requests.post(self.api_url, headers=self.headers, json=mail_payload)
            
            if mail_response.status_code != 200:
                print(f"Error fetching mail data: {mail_response.status_code}")
                print(f"Response: {mail_response.text}")
                return []
            
            mail_data = mail_response.json()
            print(f"Mail API Response: {json.dumps(mail_data, indent=2)}")
            
            if not mail_data.get("result", {}).get("success"):
                print("Failed to fetch mail data")
                print(f"Response: {mail_data}")
                return []
            
            all_emails = mail_data["result"]["data"]
            print(f"Found {len(all_emails)} total emails in mail_mail table")
            
            # Filter emails that actually match the user's email address
            matching_emails = []
            for email in all_emails:
                email_to = email.get("email_to", "")
                extracted_email = self.extract_email_from_string(email_to)
                
                if extracted_email and extracted_email.lower() == user_email.lower():
                    matching_emails.append(email)
                    print(f"Matched email: {email_to} -> {extracted_email}")
                else:
                    print(f"No match: {email_to} -> {extracted_email}")
            
            print(f"Found {len(matching_emails)} emails that match user email {user_email}")
            
            if not matching_emails:
                print(f"No matching emails found for user {user_email}")
                return []
            
            # Get corresponding message details from mail_message table
            email_details = []
            for email in matching_emails:
                mail_message_id = email.get("mail_message_id")
                if mail_message_id:
                    message_query = f"""
                    SELECT id, subject, body, email_from, date, author_id, message_type, model, record_name
                    FROM mail_message 
                    WHERE id = {mail_message_id}
                    """
                    
                    message_payload = {"query": message_query}
                    message_response = requests.post(self.api_url, headers=self.headers, json=message_payload)
                    
                    if message_response.status_code == 200:
                        message_data = message_response.json()
                        if message_data.get("result", {}).get("success") and message_data["result"]["data"]:
                            message = message_data["result"]["data"][0]
                            
                            # Combine mail and message data
                            email_detail = {
                                "mail_id": email.get("id"),
                                "message_id": mail_message_id,
                                "email_to": email.get("email_to"),
                                "subject": message.get("subject", "No Subject"),
                                "body": message.get("body", ""),
                                "email_from": message.get("email_from", ""),
                                "date": message.get("date", email.get("write_date")),
                                "message_type": message.get("message_type", ""),
                                "model": message.get("model", ""),
                                "record_name": message.get("record_name", ""),
                                "author_id": message.get("author_id"),
                                "create_date": email.get("write_date")
                            }
                            email_details.append(email_detail)
                        else:
                            print(f"No message found for mail_message_id {mail_message_id}")
                else:
                    print(f"No mail_message_id found for email {email.get('id')}")
            
            print(f"Successfully processed {len(email_details)} emails with message details")
            return email_details
            
        except Exception as e:
            print(f"Error fetching recent emails: {e}")
            return []
    
    def get_author_name(self, author_id):
        """Get author name from res_users table"""
        if not author_id:
            return "Unknown"
        
        query = f"""
        SELECT name, login
        FROM res_users 
        WHERE id = {author_id}
        """
        
        payload = {"query": query}
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                if data.get("result", {}).get("success") and data["result"]["data"]:
                    user_data = data["result"]["data"][0]
                    return user_data.get("name", user_data.get("login", "Unknown"))
                else:
                    return "Unknown"
            else:
                return "Unknown"
        except Exception as e:
            print(f"Error getting author name: {e}")
            return "Unknown"
    
    def fetch_user_emails(self, user_id):
        """Main function to fetch emails for a specific user"""
        print(f"Fetching emails for user ID: {user_id}")
        
        # Get user's email address
        user_email = self.get_user_emails(user_id)
        if not user_email:
            print(f"Could not find email for user ID {user_id}")
            return []
        
        print(f"User email: {user_email}")
        
        # Get recent emails
        emails = self.get_recent_emails(user_email)
        print(f"Emails: {emails}")
        return emails
    
    def get_emails_for_user(self, user_id):
        """Get emails for a user without printing - returns data for integration"""
        try:
            # Get user's email address
            user_email = self.get_user_emails(user_id)
            if not user_email:
                return {"error": f"Could not find email for user ID {user_id}", "emails": []}
            
            # Get recent emails
            emails = self.get_recent_emails(user_email)
            
            # Enhance emails with author names
            for email in emails:
                if email.get("author_id"):
                    email["author_name"] = self.get_author_name(email["author_id"])
                else:
                    email["author_name"] = "Unknown"
            
            return {
                "user_id": user_id,
                "user_email": user_email,
                "emails": emails,
                "count": len(emails),
            }
            
        except Exception as e:
            return {"error": str(e), "emails": []}
    
    def get_emails_clean(self, user_id):
        """Clean method to get emails for frontend integration - no printing"""
        try:
            user_email = self.get_user_emails(user_id)
            if not user_email:
                return []
            
            emails = self.get_recent_emails(user_email)
            
            # Enhance emails with author names
            for email in emails:
                if email.get("author_id"):
                    email["author_name"] = self.get_author_name(email["author_id"])
                else:
                    email["author_name"] = "Unknown"
            
            return emails
            
        except Exception as e:
            print(f"Error in get_emails_clean: {e}")
            return []
    
    def display_emails(self, emails):
        """Display emails in a formatted way"""
        if not emails:
            print("No emails to display")
            return
        
        print(f"\n{'='*80}")
        print(f"RECENT EMAILS ({len(emails)} found)")
        print(f"{'='*80}")
        
        for i, email in enumerate(emails, 1):
            print(f"\n{i}. Email ID: {email['mail_id']}")
            print(f"   Subject: {email['subject']}")
            print(f"   From: {email['email_from']}")
            print(f"   To: {email['email_to']}")
            print(f"   Date: {email['date']}")
            print(f"   {'-'*60}")
            if email['body']:
                # Truncate body if too long
                body_preview = email['body'][:200] + "..." if len(email['body']) > 200 else email['body']
                print(f"   Body Preview: {body_preview}")

def main():
    # Initialize the fetcher with your API key
    fetcher = EmailActivityFetcher()
    
    # Get user ID from input (you can modify this as needed)
    try:
        user_id = int(input("Enter user ID to fetch emails for: "))
    except ValueError:
        print("Please enter a valid user ID (number)")
        return
    
    emails = fetcher.fetch_user_emails(user_id)
    
    # Display the emails
    fetcher.display_emails(emails)
    
    # Summary
    if emails:
        print(f"\nSummary: Found {len(emails)} emails for user ID {user_id}")
        print(f"Time range: Last 2 days")
    else:
        print(f"\nNo emails found for user ID {user_id} in the last 2 days")

if __name__ == "__main__":
    main()
