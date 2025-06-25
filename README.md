# BugSight

## Setup

1. Install [uv](https://docs.astral.sh/uv/)
2. Change directory to `repstep` (the directory with `main.py`)
3. Create an `.env` file:

```
OPENAI_API_KEY={Your OpenAI API key}
```

4. Log into the [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli):

```
uv run huggingface-cli login
```

## Commands

Prepare data (download from HuggingFace, process it):
```sh
uv run main.py \
    --logs_dir results/logs \
    prepare_data \
    --dataset princeton-nlp/SWE-bench_Multimodal \
    --split dev \
    --output_file results/data/swe_df.parquet 
```

Transcribe images:
```sh
uv run main.py \
    --logs_dir results/logs \
    transcribe \
    --instances_file results/data/swe_df.parquet \
    --model gpt-4o-mini \
    --output_file results/data/swe_df_transcribed.parquet
```

Generate embeddings:
```sh
uv run main.py \
    --logs_dir results/logs \
    retrieve_swe \
    --instances_file results/data/swe_df_transcribed.parquet \
    --retrieval_field transcription \
    --testbed_dir results/testbed \
    --output_file results/retrieval.jsonl \
    --embedding_dir results/embeddings/text-embedding-3-small \
    --embedding_model text-embedding-3-small \
    --filter_model text-embedding-3-small \
    --filter_multimodal \
    --entire_file
```

Get reproductions steps: TODO

## Credits

A large part of the code is inspired by [Agentless-Lite](https://github.com/sorendunn/Agentless-Lite).
