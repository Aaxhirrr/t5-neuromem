#!/bin/bash

# create folders
mkdir -p app training memory eval infra tests

# create files with placeholders
echo "# TODO: implement" > app/server.py
echo "# TODO: implement" > app/inference.py
echo "# TODO: implement" > app/memory_retrieve.py

echo "# Colab TPU LoRA notebook scaffold will go here" > training/colab_tpu_lora.ipynb

echo "# TODO: implement" > memory/ingest.py
echo "# TODO: implement" > memory/retention_job.py

echo "# TODO: implement" > eval/run_eval.py

echo "# TODO: implement" > infra/Dockerfile
echo "{ }" > infra/devcontainer.json

echo "# TODO: implement" > tests/test_inference.py

# top-level docs
echo "# T5-NeuroMem" > README.md
echo "# Design Doc (to be filled)" > design_doc.md

echo "✅ Repo skeleton created."
