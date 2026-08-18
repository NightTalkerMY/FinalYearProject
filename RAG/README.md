# Live Retrieval Service

This directory contains the retrieval service called by the HoloPi orchestrator.

## Entry Point

Run `main.py` inside the RAG environment. The service exposes `POST /get_context` on port 8002 by default.

The live query path in `DatabaseRouting.py` performs:

1. instruction-tuned semantic routing;
2. dense top-50 retrieval from Chroma;
3. BM25 top-50 retrieval;
4. content-based deduplication;
5. standard cross-encoder prediction with batch size 20;
6. descending score sorting and thresholding.

`db/` and `teleoracle_v2_anchors.npz` are retained in Git. Transformer and embedding models are obtained from their external model sources or local caches.

## LARA Boundary

This live service is not the complete LARA implementation evaluated in the manuscript. `DatabaseRouting` retains a `use_length_sorting` constructor field, but the live query method does not use it, and reciprocal rank anchoring is absent.

The evaluated LARA implementation is in `../experiment_metric/reranker_metric/eval.ipynb`. It remains benchmark-only until it is deliberately integrated and verified. Do not describe the current live retrieval service as executing LARA.

See `../docs/reproducibility/manuscript-traceability.md` for the table-level evidence map.
