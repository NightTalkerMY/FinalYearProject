import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

import numpy as np

class ClusterSemanticRouter:
    def __init__(self, anchors_path="teleoracle_v2_anchors.npz", threshold=0.7):
        data = np.load(anchors_path)
        self.P_all = data['centroids'] 
        self.owners = data['owners']   
        self.threshold = threshold     
        
        # --- CALCULATE QUANTITATIVE BIAS ---
        unique_owners, counts = np.unique(self.owners, return_counts=True)
        total_anchors = len(self.owners)
        self.biases = {o: 1 + (1 - (c / total_anchors)) for o, c in zip(unique_owners, counts)}
                    
    def _l2_normalize(self, vector):
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)

    def _squash(self, v, eps=1e-12):
        """Capsule Squash: Maps biased energy to a squashed confidence score."""
        scale = (v**2) / (1.0 + v**2 + eps)
        return scale

    def route(self, query_embedding, iterations=2, use_bias=True, verbose=False):
        if verbose: 
            mode = "BIASED" if use_bias else "UNBIASED"
            print(f"\n{'='*20} QUANTITATIVE {mode} ROUTING {'='*20}")
        
        # 1. Normalize Query
        q = self._l2_normalize(query_embedding)
        
        # 2. Get Raw Similarities
        raw_S = np.dot(self.P_all, q) 
        
        if verbose:
            indexed_scores = sorted(zip(raw_S, self.owners), key=lambda x: x[0], reverse=True)
            print(f"[STEP 2] Top 5 Raw Matches:")
            for i, (score, owner) in enumerate(indexed_scores[:5]):
                print(f"   {i+1}. Score: {score:.4f} | DB: {owner}")

        # 3. Iterative Refinement
        refined_S = raw_S.copy()
        for i in range(iterations):
            weights = np.exp(refined_S * 10) / np.sum(np.exp(refined_S * 10))
            refined_S = raw_S * weights 

        # 4. Pick Winner with Toggleable Bias
        db_final_scores = {}
        squashed_confidences = {}

        for db_name in np.unique(self.owners):
            mask = (self.owners == db_name)
            peak_refined = np.max(refined_S[mask])
            
            # --- TOGGLE BIAS LOGIC ---
            multiplier = self.biases.get(db_name, 1.0) if use_bias else 1.0
            biased_value = peak_refined * multiplier
            db_final_scores[db_name] = biased_value
            
            squashed_confidences[db_name] = self._squash(biased_value)
            
            if verbose:
                print(f"[STEP 4] DB '{db_name}' Peak: {peak_refined:.4f} | Biased: {biased_value:.4f} (x{multiplier:.2f})")
        
        # Normalize squashed confidences
        total_conf = sum(squashed_confidences.values()) + 1e-12
        db_probabilities = {k: v / total_conf for k, v in squashed_confidences.items()}
        
        best_db = max(db_probabilities, key=db_probabilities.get)
        winning_prob = db_probabilities[best_db]

        # 5. FINAL POLICY CHECKS
        max_raw_similarity = np.max(raw_S)
        
        if verbose:
            print(f"[STEP 4.5] Winning Probability (Normalized Squash): {winning_prob:.4f}")
            print(f"[STEP 5] Final Winner Candidate: {best_db}")
            print(f"[STEP 5] Global Max Raw Similarity: {max_raw_similarity:.4f}")

        # Condition 1: Raw S Maximum check
        if max_raw_similarity < self.threshold:
            if verbose: print(f"{'='*15} RESULT: OOD (Low Raw Similarity) {'='*15}\n")
            return "OOD", max_raw_similarity

        # Condition 2: Winning Probability Squash check
        if winning_prob < self.threshold:
            if verbose: print(f"{'='*15} RESULT: OOD (Low Winning Probability) {'='*15}\n")
            return "OOD", winning_prob
            
        if verbose: print(f"{'='*15} RESULT: {best_db} {'='*15}\n")
        return best_db, winning_prob


class ProposedRouterWrapper:
    def __init__(self, cluster_router, embedding_model):
        self.router = cluster_router
        self.model = embedding_model

    def route(self, text):
        # 1. Convert text to embedding and DISABLE the progress bar
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        
        # 2. Call the original route method
        return self.router.route(embedding, verbose=False)