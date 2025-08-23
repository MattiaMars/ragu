import gradio as gr
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import os

# List your available models here
LLM_MODELS = ["mistral", "N/A", "N/A"]
EMBEDDING_MODELS = [
    r"C:\models\all-MiniLM-L6-v2"
]

def load_pdf_list(txt_path):
    pdfs = []
    # Get the directory of the txt file
    base_dir = os.path.dirname(os.path.abspath(txt_path))
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # If the path is not absolute, make it relative to the txt file
            pdf_path = line if os.path.isabs(line) else os.path.join(base_dir, line)
            if os.path.exists(pdf_path):
                pdfs.append(pdf_path)
            else:
                print(f"[WARNING] PDF not found: {pdf_path}")
    print(f"[DEBUG] Loaded {len(pdfs)} PDFs from {txt_path}")
    return pdfs

# Use a relative path for your txt file
PDF_LIST_TXT = "pdf_list.txt"
STARTUP_PDFS = load_pdf_list(PDF_LIST_TXT)

custom_prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. Use ONLY the following context to answer the user's question. "
        "If the answer is not contained in the context, say 'I don't know.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context", "question"]
)

def get_llm(llm_model_name):
    print(f"[DEBUG] Initializing LLM: {llm_model_name}")
    return Ollama(model=llm_model_name, temperature=0.0)

def local_embedding(embedding_model_name):
    print(f"[DEBUG] Initializing Embedding Model: {embedding_model_name}")
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

def document_loader(file_path):
    print(f"[DEBUG] Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    loaded_document = loader.load()
    print(f"[DEBUG] Loaded {len(loaded_document)} pages from {file_path}")
    return loaded_document

def text_splitter(data):
    print(f"[DEBUG] Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    print(f"[DEBUG] Split into {len(chunks)} chunks.")
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
    embedding_model = local_embedding(embedding_model_name)
    vectordb.add_documents(chunks)
    print(f"[DEBUG] PDF {file.name} added to vector DB.")

def retriever_qa(file, query, llm_model_name, embedding_model_name):
    global vectordb
    print(f"[DEBUG] Received query: {query}")
    # If user uploaded a new PDF, add it to the vector DB
    if file is not None:
        print(f"[DEBUG] User uploaded file: {file.name}")
        add_pdf_to_vectordb(file, embedding_model_name)
    else:
        print(f"[DEBUG] No new PDF uploaded by user.")
    llm = get_llm(llm_model_name)
    retriever_obj = vectordb.as_retriever()
    print(f"[DEBUG] Created retriever from vector DB.")

    llm_chain = LLMChain(llm=llm, prompt=custom_prompt)
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    qa = RetrievalQA(
        retriever=retriever_obj,
        combine_documents_chain=combine_documents_chain,
        return_source_documents=True
    )
    print(f"[DEBUG] Running QA chain...")
    response = qa({"query": query})
    answer = response['result']
    sources = response.get('source_documents', [])
    print(f"[DEBUG] QA chain complete. Found {len(sources)} source documents.")
    source_texts = "\n\n---\n\n".join(
        [f"Source {i+1}:\n{doc.page_content[:1000]}" for i, doc in enumerate(sources[:3])]
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

    embedding_model_dropdown = gr.Dropdown(label="Choose Embedding Model", choices=EMBEDDING_MODELS, value=EMBEDDING_MODELS[0])
    llm_model_dropdown = gr.Dropdown(label="Choose LLM Model", choices=LLM_MODELS, value=LLM_MODELS[0])
    file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")
    query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    answer_output = gr.Textbox(label="Answer")
    sources_output = gr.Textbox(label="Source Chunks Used")

    # On startup, initialize the vector DB
    gr.Button("Initialize").click(startup, inputs=[embedding_model_dropdown])

    # Main QA function
    gr.Button("Ask").click(
        retriever_qa,
        inputs=[file_input, query_input, llm_model_dropdown, embedding_model_dropdown],
        outputs=[answer_output, sources_output]
    )

rag_application.launch(server_name="0.0.0.0", server_port=7860)
