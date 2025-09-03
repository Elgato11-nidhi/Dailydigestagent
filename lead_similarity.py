# app/lead_similarity.py

import os
import sys
import requests
import chromadb
from typing import List, Dict, Optional, Iterable, Set
from dotenv import load_dotenv
from datetime import datetime
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

API_URL = "https://staging.crm.buildmapper.ai/api/v1/execute_query"
HEADERS = {
    "API-Key": os.getenv("CRM_API_KEY"),
    "Content-Type": "application/json",
}

BATCH_SIZE = 90  # tune for your infra


def fetch_data(query: str) -> List[Dict]:
    payload = {"query": query}
    try:
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("result", {}).get("data", []) or []
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] CRM API: {e}")
        return []


def _ns_id(prefix: str, raw_id: str) -> str:
    # Namespace collection IDs to avoid collisions between tables.
    return f"{prefix}_{str(raw_id).strip()}"


class LeadSimilarityAnalyzer:
    def __init__(self):
        load_dotenv()

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("❌ Missing OPENAI_API_KEY")

        # Prefer Chroma Cloud via HttpClient if CHROMADB_CLOUD_HOST is set

        self.chroma_client = None
        try:
            print(f"[INFO] Connecting to Chroma Cloud ...")
            self.chroma_client = chromadb.CloudClient(
                api_key=os.getenv('CHROMADB_API_KEY'),
                tenant='d2d08375-42ea-4bac-854b-09bac5998a24',
                database='Daily Digest'
            )
            _ = self.chroma_client.list_collections()
            print("[INFO] ✅ Chroma Cloud connected")
        except Exception as e:
            print(f"[WARNING] Cloud connect failed: {e}")

        if self.chroma_client is None:
            print("[INFO] Using local persistent ChromaDB at ./chroma_db")
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        self.embedder = OpenAIEmbeddingFunction(
            api_key=openai_api_key, model_name="text-embedding-3-small"
        )

        # Single collection for both sources, distinguished by metadata + namespaced IDs
        self.collection = self.chroma_client.get_or_create_collection(
            name="leads_collection", embedding_function=self.embedder
        )

        # Warm the in-memory cache from storage (paginated)
        self.cached_ids: Set[str] = set()
        self._preload_all_ids()

    # ---------- Storage / Cache helpers ----------

    def _preload_all_ids(self):
        """Load *all* IDs from the collection using pagination."""
        offset = 0
        loaded = 0
        while True:
            try:
                # Some versions support limit/offset; if not, this still returns something we can use
                chunk = self.collection.get(limit=BATCH_SIZE, offset=offset)
                ids = chunk.get("ids") or []
                if not ids:
                    break
                self.cached_ids.update(ids)
                loaded += len(ids)
                offset += len(ids)
                if len(ids) < BATCH_SIZE:
                    break
            except TypeError:
                # Older servers might not support offset; fall back to single fetch
                all_ = self.collection.get()
                ids = all_.get("ids") or []
                self.cached_ids.update(ids)
                loaded = len(ids)
                break
        print(f"[INFO] Cache warmup complete: {loaded} IDs")

    def _existing_ids(self, ids: Iterable[str]) -> Set[str]:
        """Return the subset of ids that already exist in the collection."""
        existing: Set[str] = set()
        ids = list(ids)
        for i in range(0, len(ids), BATCH_SIZE):
            batch = ids[i : i + BATCH_SIZE]
            try:
                got = self.collection.get(ids=batch)
                found = set(got.get("ids") or [])
                existing.update(found)
            except Exception as e:
                print(f"[WARNING] get(ids=...) failed for batch {i}: {e}")
        return existing

    def _sanitize_metadata(self, md: Dict) -> Dict:
        out = {}
        for k, v in md.items():
            if v is None:
                out[k] = ""
            else:
                out[k] = v if isinstance(v, (int, float, bool)) else str(v)
        return out

    def _add_embeddings_if_missing(
        self,
        leads: List[Dict],
        lead_type: str,     # "existing" or "new"
        user_id: int,
        id_prefix: str,     # "crm" or "pub"
    ) -> int:
        """
        Adds embeddings only for leads whose namespaced IDs are not present.
        Uses batched existence check -> zero re-embedding for existing vectors.
        """
        docs, metas, ids = [], [], []

        # Prepare namespaced IDs and filter by existence in one go
        candidate_ids = [_ns_id(id_prefix, lead.get("id", "")) for lead in leads]
        # Filter out invalid ids or missing descriptions early
        prepared = []
        for lead, nsid in zip(leads, candidate_ids):
            if not nsid or nsid.endswith("_"):
                continue
            desc = (lead.get("project_description") or "").strip()
            if not desc:
                continue
            prepared.append((lead, nsid, desc))

        if not prepared:
            return 0

        # Storage-level check (persists across runs)
        missing = set(nsid for _, nsid, _ in prepared) - self._existing_ids(
            [nsid for _, nsid, _ in prepared]
        )
        if not missing:
            return 0

        # Build final add lists only for missing
        for lead, nsid, desc in prepared:
            if nsid not in missing:
                continue
            md = self._sanitize_metadata(
                {
                    "lead_type": lead_type,       # "existing" or "new"
                    "user_id": user_id,
                    "name": lead.get("name"),
                    "create_date": lead.get("create_date"),
                    "lead_id": str(lead.get("id", "")).strip(),
                    "stored_at": datetime.now().isoformat(),
                }
            )
            ids.append(nsid)
            docs.append(desc)
            metas.append(md)

        added = 0
        # Batch add to minimize roundtrips
        for i in range(0, len(ids), BATCH_SIZE):
            try:
                j = i + BATCH_SIZE
                self.collection.add(
                    ids=ids[i:j], documents=docs[i:j], metadatas=metas[i:j]
                )
                added += (j - i)
            except Exception as e:
                print(f"[ERROR] add() failed for [{i}:{j}]: {e}")

        # Update in-memory cache (best-effort)
        self.cached_ids.update(ids)
        if added:
            print(f"[INFO] Embedded {added} new {'CRM' if id_prefix=='crm' else 'publish'} leads")
        return added

    # ---------- Public API ----------

    def find_interest_matched_new_leads(
        self,
        user_id: int,
        lead_id: Optional[str] = None,
        similarity_threshold: float = 0.7,
        max_results: int = 3,
    ) -> List[Dict]:
        """
        Suggest similar leads in lead_publish that are NOT already in user's CRM.
        - If lead_id is None: use ALL user's CRM leads as context.
        - If lead_id is provided: use only that CRM lead as context.
        Embeddings are ensured to exist but never re-computed for already-stored items.
        """

        # ---- Fetch CRM context ----
        if lead_id:
            crm_leads = fetch_data(
                f"""
                SELECT id, name, project_description, create_date, region
                FROM crm_lead
                WHERE id = {lead_id} AND user_id = {user_id}
                limit 100
                """
            )
        else:
            crm_leads = fetch_data(
                f"""
                SELECT id, name, project_description, create_date, region
                FROM crm_lead
                WHERE project_description IS NOT NULL AND user_id = {user_id}
                limit 100
                """
            )

        if not crm_leads:
            print("[WARNING] No CRM leads found for this request.")
            return []

        crm_ids = {str(r["id"]) for r in crm_leads if r.get("id") is not None}
        region = crm_leads[0].get("region")

        # ---- Fetch candidate publish leads (exclude those already in CRM) ----
        new_leads = fetch_data(
            f"""
            SELECT id, name, project_description, create_date, project_status, region
            FROM lead_publish
            WHERE project_description IS NOT NULL
              AND project_status = 'In Review'
              AND region = '{region}'
              limit 100
            """
        )
        new_leads = [r for r in new_leads if str(r.get("id")) not in crm_ids]

        print(f"[DEBUG] CRM leads in scope: {len(crm_leads)}, publish candidates: {len(new_leads)}")

        # ---- Ensure embeddings exist (no re-embedding if already present) ----
        self._add_embeddings_if_missing(crm_leads, "existing", user_id, id_prefix="crm")
        self._add_embeddings_if_missing(new_leads, "new", user_id, id_prefix="pub")

        # ---- Match each publish lead against user's CRM embeddings ----
        matches = []
        for pub in new_leads:
            desc = (pub.get("project_description") or "").strip()
            if not desc:
                continue
            try:
                res = self.collection.query(
                    query_texts=[desc],
                    n_results=max_results,
                    where={
                        "$and": [
                            {"lead_type": {"$eq": "existing"}},
                            {"user_id": {"$eq": user_id}},
                        ]
                    },
                )
            except Exception as e:
                print(f"[ERROR] Query failed for publish lead {pub.get('id')}: {e}")
                continue

            dists = (res.get("distances") or [[]])[0]
            metas = (res.get("metadatas") or [[]])[0]
            if not dists:
                continue

            # Convert cosine distance -> similarity
            best = None
            best_sim = 0.0
            for i, dist in enumerate(dists):
                try:
                    sim = 1.0 - float(dist)
                except Exception:
                    continue
                if sim >= similarity_threshold and sim > best_sim:
                    md = metas[i] if i < len(metas) else {}
                    best_sim = sim
                    best = {
                        "new_lead": pub,
                        "matched_existing_lead_id": md.get("lead_id"),
                        "similarity_score": round(sim, 3),
                    }

            if best:
                matches.append(best)

        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:max_results]


def main():
    analyzer = LeadSimilarityAnalyzer()

    if len(sys.argv) < 2:
        print("Usage: python lead_similarity.py <user_id> [<lead_id>]")
        sys.exit(1)

    user_id = int(sys.argv[1])
    lead_id = sys.argv[2] if len(sys.argv) > 2 else None

    out = analyzer.find_interest_matched_new_leads(user_id, lead_id)
    if not out:
        print("No similar leads found.")
        return

    print(f"Top {len(out)} matches:")
    for m in out:
        nl = m["new_lead"]
        print(
            f" - New Lead {nl['id']} ({nl.get('name','N/A')}) "
            f"matched with CRM Lead {m['matched_existing_lead_id']} "
            f"(Similarity {m['similarity_score']})"
        )


if __name__ == "__main__":
    main()