import os
import sys
import requests
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

load_dotenv()

API_URL = "https://staging.crm.buildmapper.ai/api/v1/execute_query"
headers = {
    "API-Key": os.getenv('CRM_API_KEY'),
    "Content-Type": "application/json"
}

client = chromadb.CloudClient(
  api_key= os.getenv('CHROMADB_API_KEY'),
  tenant='d2d08375-42ea-4bac-854b-09bac5998a24',
  database='Daily Digest'
)


def fetch_data(query: str) -> List[Dict]:
    payload = {"query": query}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get('result', {}).get('data', [])
        else:
            return []
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request error: {e}")
        return []


class LeadSimilarityAnalyzer:
    def __init__(self, api_key: str = None):
        load_dotenv()

        env_key = os.getenv('OPENAI_API_KEY')
        self.api_key = api_key or env_key
        if not self.api_key:
            raise ValueError("OpenAI API key is missing!")

        self.client = OpenAI(api_key=self.api_key)

        # Use ChromaDB Cloud client
        self.chroma_client = client
        
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name="text-embedding-ada-002"
        )

        # Get or create collection in ChromaDB Cloud
        self.collection = self.chroma_client.get_or_create_collection(
            name="leads_collection",
            embedding_function=self.embedding_fn
        )
        
        print(f"[INFO] Connected to ChromaDB Cloud collection: {self.collection.name}")

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """
        Replace None values in metadata with safe defaults.
        """
        sanitized = {}
        for k, v in metadata.items():
            if v is None:
                sanitized[k] = ""  # Could also use 0 or False depending on field type
            else:
                sanitized[k] = str(v) if not isinstance(v, (int, float, bool)) else v
        return sanitized

    def store_lead_embeddings(self, leads: List[Dict], lead_type: str, user_id: int):
        """
        Store embeddings in ChromaDB Cloud if not already stored, with safe metadata.
        """
        stored_count = 0
        skipped_count = 0
        
        for lead in leads:
            lead_id = str(lead.get("id", "")).strip()
            if not lead_id:
                print(f"[WARNING] Missing Lead ID, skipping: {lead}")
                skipped_count += 1
                continue

            # Check if already stored
            try:
                existing_doc = self.collection.get(ids=[lead_id])
                if existing_doc.get("ids"):
                    print(f"[DEBUG] Embedding already exists for Lead ID {lead_id}")
                    continue
            except Exception as e:
                print(f"[DEBUG] No existing embedding for Lead ID {lead_id}: {e}")

            desc = lead.get("project_description", "")
            if not desc.strip():
                print(f"[WARNING] No description for Lead ID {lead_id}, skipping.")
                skipped_count += 1
                continue

            print(f"[DEBUG] Storing embedding for {lead_type} lead {lead_id} (user {user_id})")
            metadata = self._sanitize_metadata({
                "lead_type": lead_type,
                "user_id": user_id,
                "name": lead.get("name"),
                "create_date": lead.get("create_date"),
                "lead_id": lead_id,
                "stored_at": datetime.now().isoformat()
            })

            try:
                self.collection.add(
                    ids=[lead_id],
                    documents=[desc],
                    metadatas=[metadata]
                )
                stored_count += 1
                print(f"[SUCCESS] Stored embedding for Lead ID {lead_id}")
            except Exception as e:
                print(f"[ERROR] Failed to store embedding for Lead ID {lead_id}: {e}")
                skipped_count += 1
        
        print(f"[INFO] Embedding storage complete: {stored_count} stored, {skipped_count} skipped")
        return stored_count

    def get_stored_leads(self, limit: int = 10):
        """
        Retrieve stored leads from ChromaDB for verification.
        """
        try:
            results = self.collection.get(limit=limit)
            return {
                "ids": results.get("ids", []),
                "documents": results.get("documents", []),
                "metadatas": results.get("metadatas", []),
                "count": len(results.get("ids", []))
            }
        except Exception as e:
            print(f"[ERROR] Failed to retrieve leads from ChromaDB: {e}")
            return {"error": str(e)}

    def search_similar_leads(self, query: str, limit: int = 5):
        """
        Search for leads similar to a given query using ChromaDB.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            return {
                "query": query,
                "results": results
            }
        except Exception as e:
            print(f"[ERROR] Failed to search leads in ChromaDB: {e}")
            return {"error": str(e)}

    def find_interest_matched_new_leads(self, user_id: int, similarity_threshold: float = 0.7) -> List[Dict]:
        # Fetch leads from CRM
        existing_leads = fetch_data(f"""
            SELECT id, name, project_description, region 
            FROM crm_lead 
            WHERE project_description IS NOT NULL AND user_id = {user_id}
        """)

        if not existing_leads:
            print("[WARNING] No existing leads found for this user.")
            return []

        region = existing_leads[0].get("region")
        new_leads = fetch_data(f"""
            SELECT id, name, project_description, create_date, user_id, region 
            FROM crm_lead 
            WHERE project_description IS NOT NULL 
              AND create_date >= NOW() - INTERVAL '100 hours' 
              AND region = '{region}'
        """)

        print(f"[DEBUG] Found {len(existing_leads)} existing leads, {len(new_leads)} new leads.")

        # Store embeddings
        self.store_lead_embeddings(existing_leads, "existing", user_id)
        self.store_lead_embeddings(new_leads, "new", user_id)

        matches = {}
        for new_lead in new_leads:
            if not new_lead.get("project_description"):
                continue

            results = self.collection.query(
                query_texts=[new_lead["project_description"]],
                n_results=3,
                where={
                    "$and": [
                        {"lead_type": {"$eq": "existing"}},
                        {"user_id": {"$eq": user_id}}
                    ]
                }
            )

            best_similarity = 0
            best_match = None

            for idx, distance in enumerate(results["distances"][0]):
                similarity = 1 - distance
                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "new_lead": new_lead,
                        "matched_existing_lead_id": results["ids"][0][idx],
                        "similarity_score": round(similarity, 3)
                    }

            if best_match:
                # Only keep the best match per new lead
                matches[new_lead["id"]] = best_match

        # Sort all matches and return top 3 distinct new leads
        unique_matches = sorted(matches.values(), key=lambda x: x["similarity_score"], reverse=True)
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

    matched_new_leads = analyzer.find_interest_matched_new_leads(user_id, similarity_threshold=0.7)

    if not matched_new_leads:
        print("No similar new leads found in the last 24 hours.")
    else:
        print(f"Top {len(matched_new_leads)} similar new leads:\n")
        for match in matched_new_leads:
            lead = match['new_lead']
            print(f" - New Lead ID: {lead['id']}, Name: {lead.get('name', 'N/A')}, "
                  f"Matched with Existing Lead ID: {match['matched_existing_lead_id']}, "
                  f"Similarity: {match['similarity_score']}")


if __name__ == "__main__":
    main()