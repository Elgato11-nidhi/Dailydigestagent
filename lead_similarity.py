# app/lead_similarity.py

import os
import sys
import requests
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma   
from chromadb.config import Settings as ChromaSettings

load_dotenv()

API_URL = "https://staging.crm.buildmapper.ai/api/v1/execute_query"
headers = {
    "API-Key": os.getenv("CRM_API_KEY"),
    "Content-Type": "application/json",
}


def fetch_data(query: str) -> List[Dict]:
    payload = {"query": query}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("result", {}).get("data", [])
        else:
            print(f"[ERROR] CRM API returned {response.status_code}: {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request error: {e}")
        return []


class LeadSimilarityAnalyzer:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY in environment variables")

        # ✅ Embedding function
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small",
        )

        # ✅ Configure Chroma Cloud client
        try:
            self.vectorstore = Chroma(
                collection_name="leads_collection",
                embedding_function=self.embeddings,
                client_settings=ChromaSettings(
                    chroma_server_host="api.trychroma.com",   # ✅ Chroma Cloud endpoint
                    chroma_server_http_port=443,
                    chroma_server_headers={
                        "Authorization": f"Bearer {os.getenv('CHROMADB_API_KEY')}"
                    }
                ),
            )
            print("[INFO] Connected to Chroma Cloud v2")
        except Exception as e:
            print(f"[WARNING] Failed to connect to Chroma Cloud: {e}")
            print("[INFO] Falling back to local ChromaDB (persistent)")
            self.vectorstore = Chroma(
                collection_name="leads_collection",
                embedding_function=self.embeddings,
                persist_directory="./chroma_db",
            )

    def test_connection(self):
        """Test ChromaDB connection and return status"""
        try:
            # Try to get collection info
            results = self.vectorstore.similarity_search("test", k=1)
            collection_info = len(results) if results else 0
            
            # Determine client type based on the settings
            if hasattr(self.vectorstore, '_client') and hasattr(self.vectorstore._client, '_identifier'):
                client_type = "ChromaDB Cloud"
            elif hasattr(self.vectorstore, '_persist_directory'):
                client_type = "ChromaDB Local"
            else:
                client_type = "ChromaDB In-Memory"
            
            return {
                "status": "connected",
                "collection_name": "leads_collection",
                "document_count": collection_info,
                "client_type": client_type,
                "api_version": "v2"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Ensure metadata values are safe for storage"""
        sanitized = {}
        for k, v in metadata.items():
            if v is None:
                sanitized[k] = ""
            else:
                sanitized[k] = str(v) if not isinstance(v, (int, float, bool)) else v
        return sanitized

    def store_lead_embeddings(self, leads: List[Dict], lead_type: str, user_id: int):
        stored_count = 0
        skipped_count = 0

        for lead in leads:
            lead_id = str(lead.get("id", "")).strip()
            if not lead_id:
                print(f"[WARNING] Missing Lead ID, skipping: {lead}")
                skipped_count += 1
                continue

            desc = lead.get("project_description", "")
            if not desc.strip():
                print(f"[WARNING] No description for Lead ID {lead_id}, skipping.")
                skipped_count += 1
                continue

            metadata = self._sanitize_metadata(
                {
                    "lead_type": lead_type,
                    "user_id": user_id,
                    "name": lead.get("name"),
                    "create_date": lead.get("create_date"),
                    "lead_id": lead_id,
                    "stored_at": datetime.now().isoformat(),
                }
            )

            try:
                self.vectorstore.add_texts(
                    texts=[desc],
                    metadatas=[metadata],
                    ids=[lead_id],
                )
                stored_count += 1
                print(f"[SUCCESS] Stored embedding for Lead ID {lead_id}")
            except Exception as e:
                print(f"[ERROR] Failed to store embedding for Lead ID {lead_id}: {e}")
                skipped_count += 1

        print(
            f"[INFO] Embedding storage complete: {stored_count} stored, {skipped_count} skipped"
        )
        return stored_count

    def get_stored_leads(self, limit: int = 10):
        try:
            # LangChain Chroma does not have a direct `.get(limit=...)`
            # Instead, we can perform a dummy similarity search to list some docs
            results = self.vectorstore.similarity_search(" ", k=limit)
            return [
                {
                    "lead_id": doc.metadata.get("lead_id"),
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                }
                for doc in results
            ]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve leads: {e}")
            return {"error": str(e)}

    def search_similar_leads(self, query: str, limit: int = 5):
        try:
            results = self.vectorstore.similarity_search(query, k=limit)
            return [
                {
                    "lead_id": doc.metadata.get("lead_id"),
                    "metadata": doc.metadata,
                    "content": doc.page_content,
                }
                for doc in results
            ]
        except Exception as e:
            print(f"[ERROR] Failed to search leads: {e}")
            return {"error": str(e)}

    def find_interest_matched_new_leads(
        self, user_id: int, similarity_threshold: float = 0.7
    ) -> List[Dict]:
        existing_leads = fetch_data(
            f"""
            SELECT id, name, project_description, region 
            FROM crm_lead 
            WHERE project_description IS NOT NULL AND user_id = {user_id}
        """
        )

        if not existing_leads:
            print("[WARNING] No existing leads found for this user.")
            return []

        region = existing_leads[0].get("region")
        new_leads = fetch_data(
            f"""
            SELECT id, name, project_description, create_date, user_id, region 
            FROM crm_lead 
            WHERE project_description IS NOT NULL 
              AND create_date >= NOW() - INTERVAL '24 hours' 
              AND region = '{region}'
        """
        )

        print(
            f"[DEBUG] Found {len(existing_leads)} existing leads, {len(new_leads)} new leads."
        )

        self.store_lead_embeddings(existing_leads, "existing", user_id)
        self.store_lead_embeddings(new_leads, "new", user_id)

        matches = {}
        for new_lead in new_leads:
            if not new_lead.get("project_description"):
                continue

            results = self.vectorstore.similarity_search_with_score(
                new_lead["project_description"], k=3
            )

            best_match = None
            best_similarity = 0
            for doc, score in results:
                # Score is distance; lower = more similar
                similarity = 1 - score
                if (
                    similarity >= similarity_threshold
                    and similarity > best_similarity
                    and doc.metadata.get("lead_type") == "existing"
                    and int(doc.metadata.get("user_id", -1)) == user_id
                ):
                    best_similarity = similarity
                    best_match = {
                        "new_lead": new_lead,
                        "matched_existing_lead_id": doc.metadata.get("lead_id"),
                        "similarity_score": round(similarity, 3),
                    }

            if best_match:
                matches[new_lead["id"]] = best_match

        unique_matches = sorted(
            matches.values(), key=lambda x: x["similarity_score"], reverse=True
        )
        return unique_matches[:3]


def main():
    load_dotenv()
    analyzer = LeadSimilarityAnalyzer()

    if len(sys.argv) < 2:
        print("Usage: python lead_similarity.py <user_id>")
        sys.exit(1)

    try:
        user_id = int(sys.argv[1])
    except ValueError:
        print("Error: User ID must be an integer.")
        sys.exit(1)

    matched_new_leads = analyzer.find_interest_matched_new_leads(
        user_id, similarity_threshold=0.7
    )

    if not matched_new_leads:
        print("No similar new leads found in the last 24 hours.")
    else:
        print(f"Top {len(matched_new_leads)} similar new leads:\n")
        for match in matched_new_leads:
            lead = match["new_lead"]
            print(
                f" - New Lead ID: {lead['id']}, Name: {lead.get('name', 'N/A')}, "
                f"Matched with Existing Lead ID: {match['matched_existing_lead_id']}, "
                f"Similarity: {match['similarity_score']}"
            )


if __name__ == "__main__":
    main()