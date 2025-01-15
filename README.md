# Introduction

this project is to generate reports with KG extracted from patents

# Usage

## Install prerequisites

```shell
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
docker run --gpus all --shm-size 1g -p 8080:80 -v <home>/.cache/huggingface:/data ghcr.io/huggingface/text-generation-inference --model-id $model
```

