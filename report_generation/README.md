# Introduction

this project is to generate reports with KG extracted from patents

# Usage

## Install prerequisites

```shell
sudo apt install poppler-utils tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-chi-sim-vert tesseract-ocr-chi-tra-vert tesseract-ocr-script-hang-vert tesseract-ocr-script-hang tesseract-ocr-script-hans-vert tesseract-ocr-script-hans tesseract-ocr-script-hant-vert tesseract-ocr-script-hant
python3 -m pip install -r requirements.txt
```

## Launch Neo4j

```shell
docker pull neo4j:enterprise-bullseye
docker run -d --publish=7474:7474 --publish=7687:7687 \
           --volume=$HOME/neo4j/data:/data \
           --name neo4j-apoc \
           -e NEO4J_apoc_export_file_enabled=true \
           -e NEO4J_apoc_import_file_enabled=true \
           -e NEO4J_apoc_import_file_use__neo4j__config=true \
           -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
           --privileged --shm-size 12G -e NEO4J_ACCEPT_LICENSE_AGREEMENT=yes --cpus=32 --memory=128G neo4j:enterprise-bullseye
```

create a database for receiving graph database. if you use default config create a database of name "neo4j".

## Deploy text-generate-inference

```shell
model=Qwen/Qwen2.5-7B-Instruct
docker pull ghcr.io/huggingface/text-generation-inference
docker run --gpus all --shm-size 1g -e CUDA_VISIBLE_DEVICES=4,5,6,7 -p 8080:80 -v <home>/.cache/huggingface:/data ghcr.io/huggingface/text-generation-inference --model-id $model --max-input-length 52207 --max-batch-prefill-tokens 52207 --max-total-tokens 131072 --max-batch-size 32 --num-shard 4
```

## Load documents into graph database

```shell
python3 load_graphdb.py --input_dir patents
```

## Start patent report service

```shell
python3 main.py
```

try to ask agent a question such as "Help me generate a report analyzing the LoftQ experimental techniques" to start the report assembling process.
