# ReDynToT

This repository contains the ReDynToT method code, the prompt templates used by its three dataset profiles, and a 600-example decomposition pool. The default configuration uses five exemplars on HotpotQA, 2WikiMQA, and MuSiQue, matching the main comparison in the revised manuscript.

## Repository contents

- `src/redyntot/`: question decomposition, exemplar matching, selective reconstruction, tree execution, retrieval, and answer aggregation.
- `prompts/`: dataset-specific prompt templates.
- `configs/`: dataset-specific runtime settings, including the model identifier, exemplar count, reconstruction threshold, retrieval limits, and index name.
- `data/exemplar_pool_600.json`: the 600-example pool.
- `tests/`: offline checks that do not call a model or retrieval service.
- `docs/`: method stages and data setup.

Benchmark datasets, retrieval corpora and indexes, model weights, and the paper's evaluation inputs and outputs are not bundled. Obtain the datasets from their original sources and prepare the retrieval resources described in `docs/DATA_AND_RETRIEVAL.md`. The repository alone does not regenerate the manuscript tables.

## Setup

Use Python 3.10 or later. In a fresh environment, run:

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The dependency ranges are not an exact environment lockfile. Record installed package and model versions for a new run.

Run commands from the repository root. Set `PYTHONPATH=src` on Linux/macOS or `$env:PYTHONPATH = (Join-Path (Get-Location) 'src')` in PowerShell. Then check the command-line interface and run the offline tests:

```bash
python -m redyntot --help
python -m unittest discover -s tests -v
```

## Running the method

Prepare a JSONL file containing a stable `id`, `question`, and gold `answer` for each item. The stages write new output files and do not resume an interrupted run.

```bash
python -m redyntot build-prompts --questions questions.jsonl --config configs/hotpotqa.json --output outputs/prompts.jsonl
python -m redyntot decompose --prompts outputs/prompts.jsonl --config configs/hotpotqa.json --output outputs/decompositions.jsonl
python -m redyntot prepare-trees --input outputs/decompositions.jsonl --profile hotpotqa --output outputs/trees.jsonl
python -m redyntot answer --trees outputs/trees.jsonl --config configs/hotpotqa.json --corpus corpora/hotpotqa_wikipedia.jsonl --index-dir indexes/hotpotqa --output outputs/answers.jsonl
python -m redyntot evaluate --questions questions.jsonl --predictions outputs/answers.jsonl --output outputs/metrics.json
```

Use the corresponding configuration and corpus for 2WikiMQA or MuSiQue. Corpus preparation, Elasticsearch indexing, dense-index construction, and model-weight downloads are separate setup steps; see `docs/DATA_AND_RETRIEVAL.md`.

Set `OPENAI_API_KEY` in the environment. The default endpoint uses the OpenAI chat-completions API. `OPENAI_BASE_URL` sets a base URL; `REDYNTOT_CHAT_URL` overrides the full endpoint and `REDYNTOT_API_KEY` overrides the key. Configure Elasticsearch with `REDYNTOT_ES_URL` and, if needed, `REDYNTOT_ES_USER`, `REDYNTOT_ES_PASSWORD`, and `REDYNTOT_ES_CA_CERT`. No credentials are included.

The selected model endpoint must return generated-token log probabilities in `choices[0].logprobs.content`. A model name in a configuration file is the requested identifier; check the returned-model field in the run record to establish which model served the request.

The exemplar pool contains material derived from the three benchmarks. See `THIRD_PARTY_NOTICES.md` and the original dataset sources before redistributing it.
