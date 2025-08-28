# app/lead_similarity.py

import os
import sys
import requests
import chromadb
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

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

        # ✅ OpenAI client for embeddings
        self.client = OpenAI(api_key=api_key)

        # ✅ Configure Chroma Cloud client
        try:
            self.chroma_client = chromadb.CloudClient(
                api_key=os.getenv('CHROMADB_API_KEY'),
                tenant='d2d08375-42ea-4bac-854b-09bac5998a24',
                database='Daily Digest'
            )
            
            # Create or get collection with OpenAI embedding function
            self.collection = self.chroma_client.get_or_create_collection(
                name="leads_collection",
                embedding_function=OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-3-small"
                )
            )
            print("[INFO] Connected to ChromaDB Cloud")
        except Exception as e:
            print(f"[WARNING] Failed to connect to ChromaDB Cloud: {e}")
            print("[INFO] Falling back to local ChromaDB (persistent)")
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                name="leads_collection",
                embedding_function=OpenAIEmbeddingFunction(
                    api_key=api_key,
                    model_name="text-embedding-3-small"
                )
            )

    def test_connection(self):
        """Test ChromaDB connection and return status"""
        try:
            # Try to get collection info
            collection_info = self.collection.count()
            
            # Determine client type
            if hasattr(self.chroma_client, '_identifier'):
                client_type = "ChromaDB Cloud"
            elif hasattr(self.chroma_client, '_path'):
                client_type = "ChromaDB Local"
            else:
                client_type = "ChromaDB In-Memory"
            
            return {
                "status": "connected",
                "collection_name": "leads_collection",
                "document_count": collection_info,
                "client_type": client_type,
                "api_version": "v1",
                "storage_type": "Permanent Cloud Storage" if client_type == "ChromaDB Cloud" else "Local Storage"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def verify_permanent_storage(self):
        """Verify that embeddings are stored permanently and not duplicated"""
        try:
            # Get all stored embeddings
            all_results = self.collection.get()
            
            # Check for duplicates by lead_id
            lead_ids = []
            duplicates = []
            for lead_id in all_results['ids']:
                if lead_id in lead_ids:
                    duplicates.append(lead_id)
                else:
                    lead_ids.append(lead_id)
            
            # Group by lead_type and user_id
            lead_types = {}
            user_ids = {}
            for i, metadata in enumerate(all_results['metadatas']):
                lead_type = metadata.get("lead_type", "unknown")
                user_id = metadata.get("user_id", "unknown")
                
                lead_types[lead_type] = lead_types.get(lead_type, 0) + 1
                user_ids[user_id] = user_ids.get(user_id, 0) + 1
            
            return {
                "status": "success",
                "total_embeddings": len(all_results['ids']),
                "unique_lead_ids": len(lead_ids),
                "duplicate_lead_ids": len(duplicates),
                "duplicates_found": duplicates[:10],  # Show first 10 duplicates
                "lead_type_distribution": lead_types,
                "user_id_distribution": user_ids,
                "storage_verified": len(duplicates) == 0
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

            # Check if already stored to avoid duplicates
            try:
                # Search for existing embedding by lead_id in metadata
                existing_results = self.collection.get(
                    where={"lead_id": lead_id}
                )
                if existing_results['ids']:
                    print(f"[DEBUG] Embedding already exists for Lead ID {lead_id}, skipping...")
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"[DEBUG] Could not check for existing embedding for Lead ID {lead_id}: {e}")

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
                # Store embedding permanently in ChromaDB Cloud
                self.collection.add(
                    documents=[desc],
                    metadatas=[metadata],
                    ids=[lead_id],
                )
                stored_count += 1
                print(f"[SUCCESS] Stored NEW embedding for Lead ID {lead_id} in ChromaDB Cloud")
            except Exception as e:
                print(f"[ERROR] Failed to store embedding for Lead ID {lead_id}: {e}")
                skipped_count += 1

        print(
            f"[INFO] Embedding storage complete: {stored_count} NEW stored, {skipped_count} already existed/skipped"
        )
        return stored_count

    def get_stored_leads(self, limit: int = 10):
        try:
            # Get all stored leads
            all_results = self.collection.get()
            
            # Limit the results
            limited_results = {
                'ids': all_results['ids'][:limit],
                'metadatas': all_results['metadatas'][:limit],
                'documents': all_results['documents'][:limit] if 'documents' in all_results else []
            }
            
            return [
                {
                    "lead_id": lead_id,
                    "metadata": metadata,
                    "content": document if 'documents' in all_results else "",
                }
                for lead_id, metadata, document in zip(
                    limited_results['ids'], 
                    limited_results['metadatas'], 
                    limited_results.get('documents', [''] * len(limited_results['ids']))
                )
            ]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve leads: {e}")
            return {"error": str(e)}

    def search_similar_leads(self, query: str, limit: int = 5):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            return [
                {
                    "lead_id": lead_id,
                    "metadata": metadata,
                    "content": document if 'documents' in results else "",
                }
                for lead_id, metadata, document in zip(
                    results['ids'][0], 
                    results['metadatas'][0], 
                    results.get('documents', [[''] * len(results['ids'][0])])[0]
                )
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

            results = self.collection.query(
                query_texts=[new_lead["project_description"]],
                n_results=3,
                where={"lead_type": "existing", "user_id": user_id}
            )

            best_match = None
            best_similarity = 0
            for i, score in enumerate(results['distances'][0]):
                # Score is distance; lower = more similar
                similarity = 1 - score
                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "new_lead": new_lead,
                        "matched_existing_lead_id": results['metadatas'][0][i].get("lead_id"),
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