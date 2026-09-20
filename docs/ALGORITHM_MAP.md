# Method stages

| Stage | Main modules | Function |
| --- | --- | --- |
| Exemplar selection | `profiles/*/matching.py` | Match syntactic fingerprints and select the configured number of exemplars. |
| Decomposition and reconstruction | `profiles/*/decomposition.py`, `runtime.py` | Generate a decomposition, evaluate confidence and structural triggers, and reconstruct flagged substructures. |
| Reference correction and tree preparation | `profiles/*/references.py`, `profiles/*/tree.py` | Resolve placeholders and construct a post-order executable tree. |
| Retrieval and reranking | `profiles/*/dense.py`, `profiles/*/rerank.py`, `sparse.py` | Combine dense and sparse results and rerank passages. |
| Node answering | `profiles/*/node_evidence_answer.py`, `profiles/*/node_context_answer.py` | Answer nodes using retrieved evidence and resolved upstream answers. |
| Aggregation | `profiles/*/execution.py` | Aggregate child answers and arbitrate against direct retrieval using the configured within-node margin. |

Each dataset profile keeps its own parsing, prompts, retrieval settings, and execution code. The shared `openai_client.py` module provides the model transport. Baseline methods reported in the paper are not implemented in this repository.
