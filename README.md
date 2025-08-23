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

.
PDF not loading: Ensure your PDF is not encrypted or corrupted.
License
MIT License

