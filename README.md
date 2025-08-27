Ollama PDF RAG Chatbot
This project is a local Retrieval-Augmented Generation (RAG) chatbot that answers questions using the content of uploaded PDF documents. It uses Ollama to run a local Large Language Model (LLM) (such as Mistral, Llama 2, etc.) and LangChain for document processing and retrieval. Embeddings are generated using SentenceTransformers models.

Features
Private & Local: All processing and inference happens on your machine.
RAG Pipeline: Answers are grounded in the content of your uploaded PDF.
Customizable LLM: Easily switch between different Ollama models (Mistral, Llama 2, etc.).
User-friendly UI: Simple Gradio web interface for uploading PDFs and asking questions.
System Prompt: The assistant only answers using the provided document context and says "I don't know" if the answer is not found.
How It Works
Upload a PDF via the web interface.
The PDF is split into text chunks and embedded using a SentenceTransformers model.
Your question is embedded and used to retrieve the most relevant chunks.
The selected context and your question are sent to the local LLM (via Ollama) with a system prompt.
The LLM generates an answer based only on the retrieved context.
Requirements
Python 3.8+
Ollama installed and running locally
At least one Ollama model pulled (e.g.
ollama pull mistral

)
Python packages:
langchain

langchain-community

chromadb

gradio

transformers

sentence-transformers

pypdf

Install dependencies with:




pip install langchain langchain-community chromadb gradio transformers sentence-transformers pypdf

Usage
Start Ollama (in a terminal):




ollama serve
# or just
ollama

Pull a model (if you haven't already):




ollama pull mistral
# or
ollama pull llama2

Run the chatbot:




python ragu.py

Open your browser and go to http://localhost:7860.

Upload a PDF and ask questions!

Customization
Change the LLM:
In
get_llm()

, set
model="your_model_name"

to use a different Ollama model.
Change the embedding model:
In
local_embedding()

, set
model_name

to any SentenceTransformers model.
Modify the system prompt:
Edit the
custom_prompt

in the code to change the assistant’s behavior.
Security & Privacy
All data and models are processed locally.
No data is sent to external servers.
Troubleshooting
Ollama not found: Make sure Ollama is installed and running.
Model not found: Pull the desired model with
ollama pull modelname





Working version examples:

Name: langchain
Version: 0.3.27
Summary: Building applications with LLMs through composability
Home-page:
Author:
Author-email:
License: MIT
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires: langchain-core, langchain-text-splitters, langsmith, pydantic, PyYAML, requests, SQLAlchemy
Required-by: langchain-community
---
Name: langchain-community
Version: 0.3.27
Summary: Community contributed LangChain integrations.
Home-page:
Author:
Author-email:
License: MIT
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires: aiohttp, dataclasses-json, httpx-sse, langchain, langchain-core, langsmith, numpy, pydantic-settings, PyYAML, requests, SQLAlchemy, tenacity
Required-by:
---
Name: chromadb
Version: 1.0.20
Summary: Chroma.
Home-page: https://github.com/chroma-core/chroma
Author:
Author-email: Jeff Huber <jeff@trychroma.com>, Anton Troynikov <anton@trychroma.com>
License:
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires: bcrypt, build, grpcio, httpx, importlib-resources, jsonschema, kubernetes, mmh3, numpy, onnxruntime, opentelemetry-api, opentelemetry-exporter-otlp-proto-grpc, opentelemetry-sdk, orjson, overrides, posthog, pybase64, pydantic, pypika, pyyaml, rich, tenacity, tokenizers, tqdm, typer, typing-extensions, uvicorn
Required-by:
---
Name: gradio
Version: 5.43.1
Summary: Python library for easily interacting with trained machine learning models
Home-page: https://github.com/gradio-app/gradio
Author:
Author-email: Abubakar Abid <gradio-team@huggingface.co>, Ali Abid <gradio-team@huggingface.co>, Ali Abdalla <gradio-team@huggingface.co>, Dawood Khan <gradio-team@huggingface.co>, Ahsen Khaliq <gradio-team@huggingface.co>, Pete Allen <gradio-team@huggingface.co>, Ömer Faruk Özdemir <gradio-team@huggingface.co>, Freddy A Boulton <gradio-team@huggingface.co>, Hannah Blair <gradio-team@huggingface.co>
License-Expression: Apache-2.0
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires: aiofiles, anyio, brotli, fastapi, ffmpy, gradio-client, groovy, httpx, huggingface-hub, jinja2, markupsafe, numpy, orjson, packaging, pandas, pillow, pydantic, pydub, python-multipart, pyyaml, ruff, safehttpx, semantic-version, starlette, tomlkit, typer, typing-extensions, uvicorn
Required-by:
---
Name: transformers
Version: 4.55.4
Summary: State-of-the-art Machine Learning for JAX, PyTorch and TensorFlow
Home-page: https://github.com/huggingface/transformers
Author: The Hugging Face team (past and future) with the help of all our contributors (https://github.com/huggingface/transformers/graphs/contributors)
Author-email: transformers@huggingface.co
License: Apache 2.0 License
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires: filelock, huggingface-hub, numpy, packaging, pyyaml, regex, requests, safetensors, tokenizers, tqdm
Required-by: sentence-transformers
---
Name: sentence-transformers
Version: 5.1.0
Summary: Embeddings, Retrieval, and Reranking
Home-page: https://www.SBERT.net
Author:
Author-email: Nils Reimers <info@nils-reimers.de>, Tom Aarsen <tom.aarsen@huggingface.co>
License: Apache 2.0
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires: huggingface-hub, Pillow, scikit-learn, scipy, torch, tqdm, transformers, typing_extensions
Required-by:
---
Name: pypdf
Version: 6.0.0
Summary: A pure-python PDF library capable of splitting, merging, cropping, and transforming PDF files
Home-page:
Author:
Author-email: Mathieu Fenniak <biziqe@mathieu.fenniak.net>
License-Expression: BSD-3-Clause
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires:
Required-by:
---
Name: torch
Version: 2.8.0
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3-Clause
Location: C:\Users\mamarsetti\Dati\ragu\my_env\Lib\site-packages
Requires: filelock, fsspec, jinja2, networkx, sympy, typing-extensions
Required-by: sentence-transformers

(my_env) C:\Users\mamarsetti\Dati\ragu>

.
PDF not loading: Ensure your PDF is not encrypted or corrupted.
License
MIT License

