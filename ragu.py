import gradio as gr
from langchain_community.llms import Ollama, HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.runnables import RunnableSequence
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os

# List your available models here
LLM_MODELS = [
    "mistral",  # Ollama
    "Mistral-7B-Instruct-v0.3",  # Local HuggingFace
]
EMBEDDING_MODELS = [
    r"C:\models\all-MiniLM-L6-v2",
    r"C:\Models\multi-qa-MiniLM-L6-cos-v1"
]
def extract_answer(text):
    # Find the last occurrence of "Answer:" and return everything after it
    if "Answer:" in text:
        return text.split("Answer:")[-1].strip()
    return text.strip()
    
def load_pdf_list(txt_path):
    pdfs = []
    base_dir = os.path.dirname(os.path.abspath(txt_path))
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pdf_path = line if os.path.isabs(line) else os.path.join(base_dir, line)
            if os.path.exists(pdf_path):
                pdfs.append(pdf_path)
            else:
                print(f"[WARNING] PDF not found: {pdf_path}")
    print(f"[DEBUG] Loaded {len(pdfs)} PDFs from {txt_path}")
    return pdfs

PDF_LIST_TXT = "pdf_list.txt"
STARTUP_PDFS = load_pdf_list(PDF_LIST_TXT)

custom_prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. Use ONLY the following context to answer the user's question. "
        "If the answer is not contained in the context, say 'I don't know.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n"
        "Answer:"
    ),
    input_variables=["context", "input"]
)

def get_llm(llm_model_name):
    print(f"[DEBUG] Initializing LLM: {llm_model_name}")
    if llm_model_name == "mistral":
        return Ollama(model=llm_model_name, temperature=0.0)
    else:
        # Local HuggingFace model
        model_id = r"C:\Models\Mistral-7B-Instruct-v0.3"  # Change to your local model path
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=pipe)

def local_embedding(embedding_model_name):
    print(f"[DEBUG] Initializing Embedding Model: {embedding_model_name}")
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

def document_loader(file_path):
    print(f"[DEBUG] Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    for doc in loaded_document:
        doc.metadata["source_file"] = os.path.basename(file_path)
    print(f"[DEBUG] Loaded {len(loaded_document)} pages from {file_path}")
    return loaded_document

def text_splitter(docs):
    print(f"[DEBUG] Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[DEBUG] Split into {len(chunks)} chunks.")
    for chunk in chunks:
        if "source_file" not in chunk.metadata and "source_file" in docs[0].metadata:
            chunk.metadata["source_file"] = docs[0].metadata["source_file"]
    return chunks

# --- Global vector DB ---
vectordb = None

def initialize_vectordb(embedding_model_name):
    global vectordb
    print(f"[DEBUG] Initializing vector DB with embedding model: {embedding_model_name}")
    all_chunks = []
    for pdf_path in STARTUP_PDFS:
        if os.path.exists(pdf_path):
            print(f"[DEBUG] Found startup PDF: {pdf_path}")
            docs = document_loader(pdf_path)
            chunks = text_splitter(docs)
            all_chunks.extend(chunks)
        else:
            print(f"[WARNING] Startup PDF not found: {pdf_path}")
    embedding_model = local_embedding(embedding_model_name)
    vectordb = Chroma.from_documents(all_chunks, embedding_model)
    print(f"[DEBUG] Vector DB initialized with {len(all_chunks)} chunks.")

def add_pdf_to_vectordb(file, embedding_model_name):
    global vectordb
    print(f"[DEBUG] Adding user-uploaded PDF to vector DB: {file.name}")
    docs = document_loader(file.name)
    chunks = text_splitter(docs)
    print(f"[DEBUG] Adding {len(chunks)} new chunks to vector DB.")
    vectordb.add_documents(chunks)
    print(f"[DEBUG] PDF {file.name} added to vector DB.")


from langchain.chains.combine_documents import create_stuff_documents_chain

def retriever_qa(file, query, llm_model_name, embedding_model_name, sources_num):
    global vectordb
    print(f"[DEBUG] Received query: {query}")
    if file is not None:
        print(f"[DEBUG] User uploaded file: {file.name}")
        add_pdf_to_vectordb(file, embedding_model_name)
    else:
        print(f"[DEBUG] No new PDF uploaded by user.")
    llm = get_llm(llm_model_name)
    retriever_obj = vectordb.as_retriever(search_kwargs={"k": int(sources_num)})
    print(f"[DEBUG] Created retriever from vector DB.")

    # Use the new factory function
    combine_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=custom_prompt,
        document_variable_name="context"
    )
    retrieval_chain = create_retrieval_chain(
        combine_docs_chain=combine_documents_chain,
        retriever=retriever_obj
    )
    print(f"[DEBUG] Running retrieval chain...")
    response = retrieval_chain.invoke({"input": query})
    raw_answer = response.get("answer") or response.get("result")
    answer = extract_answer(raw_answer)
    sources = response.get("source_documents") or response.get("context") or []
    print(f"[DEBUG] Retrieval chain complete. Found {len(sources)} source documents.")
    source_texts = "\n\n---\n\n".join(
        [
            f"Source {i+1} (File: {doc.metadata.get('source_file', 'unknown')}):\n{doc.page_content[:1000]}"
            for i, doc in enumerate(sources)
        ]
    ) if sources else "No sources found."
    return answer, source_texts


# --- Gradio UI ---
def startup(embedding_model_name):
    print(f"[DEBUG] Gradio startup: initializing vector DB.")
    initialize_vectordb(embedding_model_name)
    print(f"[DEBUG] Gradio startup complete.")

with gr.Blocks() as rag_application:
    gr.Markdown("# RAG-u")
    gr.Markdown("Upload a PDF, select models, and ask any question. The chatbot will answer using all loaded documents and show the source text.")

    embedding_model_dropdown = gr.Dropdown(label="[OPTIONS] Choose Embedding Model", choices=EMBEDDING_MODELS, value=EMBEDDING_MODELS[0])
    llm_model_dropdown = gr.Dropdown(label="[OPTIONS] Choose LLM Model", choices=LLM_MODELS, value=LLM_MODELS[0])
    file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")
    query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    answer_output = gr.Textbox(label="Answer")
    sources_output = gr.Textbox(label="Source Chunks Used")
    sources_num = gr.Slider(label="[OPTIONS] Set number of sources considered", minimum=1, value=5, maximum=100, step=1)

    gr.Button("Initialize").click(startup, inputs=[embedding_model_dropdown])

    gr.Button("Ask").click(
        retriever_qa,
        inputs=[file_input, query_input, llm_model_dropdown, embedding_model_dropdown, sources_num],
        outputs=[answer_output, sources_output]
    )

rag_application.launch(server_name="10.100.63.38", server_port=7862)
