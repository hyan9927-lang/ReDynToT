# Data and retrieval setup

Download the benchmarks from their original projects:

- [HotpotQA](https://hotpotqa.github.io/)
- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
- [MuSiQue](https://github.com/StonyBrookNLP/musique)

Supply a JSONL question file with stable `id`, `question`, and `answer` fields. For example:

```json
{"id":"example-1","question":"Which city is the capital of France?","answer":"Paris"}
```

The benchmark files, retrieval corpora, dense embeddings, FAISS indexes, Elasticsearch indexes, and model weights are not included. A dense index directory must contain `embeddings.npy` and either `index.faiss` or `faiss_index.idx`; the runtime does not build a missing index. Corpus, embeddings, and index must refer to the same passages and model version.

The sparse retriever expects Elasticsearch documents with `title` and `text`. The HotpotQA corpus loader accepts JSONL paragraphs with `id`, `title`, and `text`. For 2WikiMQA and MuSiQue, follow the corresponding `profiles/*/dense.py` loader and `configs/*.json` settings. The configured Elasticsearch index names must match the local service.
