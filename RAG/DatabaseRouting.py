import chromadb
import numpy as np
import time
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

class DatabaseRouting:
    def __init__(self, db_path="db", verbose=False, use_length_sorting=True):
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.verbose = verbose
        self.use_length_sorting = use_length_sorting    

        self.product_prefix = "Product information for users looking for or interested in "
        
        # Define Embedding model
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = self.embed_model
        Settings.llm = None
        
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

    def _log(self, message):
        if self.verbose:
            print(f"[VERBOSE] {message}")

    # --- NEW: Direct ASIN Lookup (Replaces product.json) ---
    def get_product_by_asin(self, asin: str) -> str:
        """Fetch specific product content by ASIN for Gesture Exit context."""
        try:
            collection = self.db_client.get_collection("product")
            # Query metadata for exact ASIN match
            result = collection.get(where={"asins": asin}, limit=1)
            
            if result["documents"] and len(result["documents"]) > 0:
                return result["documents"][0]
            
            return "Product details not found."
        except Exception as e:
            self._log(f"Error fetching ASIN {asin}: {e}")
            return "Product details not found."

    def _strip_product_prefix(self, text: str) -> str:
        if text.startswith(self.product_prefix):
            return text[len(self.product_prefix):].strip()
        return text

    def _dedupe_leading_name(self, text: str) -> str:
        s = text.strip()
        dot_pos = s.find(". ")
        if dot_pos == -1: return s
        first = s[:dot_pos].strip()
        rest = s[dot_pos + 2 :].lstrip()
        expected_prefix = f"The {first}"
        if rest.startswith(expected_prefix):
            return first + rest[len(expected_prefix):]
        return s

    def format_product_list(self, results, strip_prefix: bool = True, max_items: int = 5) -> str:
        """
        Modified to handle the new result format (list of dicts).
        """
        cleaned = []
        # results is now a list of {"content": "...", "asin": "..."}
        for item in results[:max_items]:
            text = item["content"]
            if strip_prefix:
                text = self._strip_product_prefix(text)
            text = self._dedupe_leading_name(text)
            cleaned.append(text)

        return "\n".join(f"{i}. {doc}" for i, doc in enumerate(cleaned, start=1))

    # --- UPDATED: Return Content + Metadata ---
    def _get_vector_results(self, collection, query):
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        
        retriever = VectorIndexRetriever(index=index, similarity_top_k=50)
        nodes = retriever.retrieve(query)
        
        # Pack content AND asin into a dict
        results = []
        for n in nodes:
            results.append({
                "content": n.node.text,
                "asin": n.node.metadata.get("asins", None)
            })
            
        self._log(f"Vector Retrieval: Found {len(results)} docs.")
        return results

    def _get_bm25_results(self, collection, query):
        all_docs = collection.get() # Warning: heavy if DB is huge
        if not all_docs['documents']:
            return []
            
        # Pass metadata to Document so we don't lose the ASIN
        langchain_docs = []
        for text, meta in zip(all_docs['documents'], all_docs['metadatas']):
            langchain_docs.append(Document(page_content=text, metadata=meta))

        bm25_retriever = BM25Retriever.from_documents(documents=langchain_docs, k=50)
        results = bm25_retriever.invoke(query)
        
        # Pack content AND asin
        packed_results = []
        for d in results:
            packed_results.append({
                "content": d.page_content,
                "asin": d.metadata.get("asins", None)
            })
            
        self._log(f"BM25 Retrieval: Found {len(packed_results)} docs.")
        return packed_results

    def query(self, user_query, database_name=None):
        self._log("-" * 30)
        
        # 1. Routing
        target_db = database_name
        if target_db is None and getattr(self, "semantic_router", None):
            predicted_db, confidence = self.semantic_router.route(user_query)
            if predicted_db == "OOD":
                return [] # Return empty list, not string
            target_db = predicted_db

        try:
            collection = self.db_client.get_collection(target_db)
        except Exception:
            return []

        # 2. Retrieval (Now returns lists of dicts)
        vector_res = self._get_vector_results(collection, user_query)
        keyword_res = self._get_bm25_results(collection, user_query)
        
        # 3. Fusion (Deduplicate based on content)
        seen_content = set()
        fused_docs = []
        
        for item in vector_res + keyword_res:
            if item["content"] not in seen_content:
                fused_docs.append(item)
                seen_content.add(item["content"])

        if not fused_docs:
            return []

        # 4. Reranking Setup
        # Extract just the text for the reranker
        docs_text_only = [d["content"] for d in fused_docs]

        self._log(f"Starting Reranking on {len(fused_docs)} documents...")

        # (Simplified Reranking Logic for brevity - keeping your core logic)
        pairs = [(user_query, doc_text) for doc_text in docs_text_only]
        
        # Predict scores
        logits = self.reranker.predict(pairs, batch_size=20, show_progress_bar=False)
        
        # 5. Sort based on scores
        # Zip the FULL dictionary (with ASIN) with the score
        scored_docs = sorted(
            zip(fused_docs, logits),
            key=lambda x: x[1],
            reverse=True,
        )

        # passed_docs = []
        # self._log("Top Scored Documents:")
        # for item, logit in scored_docs:
        #     if logit >= -2:
        #         # item is {"content": "...", "asin": "..."}
        #         passed_docs.append(item)

        # passed_docs = []
        # self._log(f"Top Scored Documents (total={len(scored_docs)}):")

        # for rank, (item, logit) in enumerate(scored_docs, start=1):
        #     preview = item["content"].replace("\n", " ")[:120]
        #     self._log(f"#{rank} score={float(logit):.4f} asin={item['asin']} | {preview}")

        #     if logit >= -2:
        #         passed_docs.append(item)

        # # 6. Return Logic
        # if target_db == "product":
        #     return passed_docs[:5] # Return top 5 dicts
        # elif target_db == "retail_qna":
        #     return passed_docs[:1] if passed_docs else []

        # return passed_docs


        passed_docs = []
        self._log(f"Top Scored Documents (total={len(scored_docs)}):")

        for rank, (item, logit) in enumerate(scored_docs, start=1):
            preview = item["content"].replace("\n", " ")[:120]
            self._log(f"#{rank} score={float(logit):.4f} asin={item['asin']} | {preview}")

            if logit >= -2:
                passed_docs.append(item)

        # --- Fallback if nothing passed threshold ---
        if not passed_docs:
            self._log("No documents passed threshold. Returning N/A.")

            passed_docs = [{
                "content": "N/A",
                "asin": "N/A"
            }]

        # --- Return Logic ---
        if target_db == "product":
            return passed_docs[:5]
        elif target_db == "retail_qna":
            return passed_docs[:1]

        return passed_docs
