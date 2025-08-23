from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# ---- LLM setup (Ollama) ----
def get_llm():
    # Use the model you pulled with ollama, e.g. "mistral", "llama2", "phi3", etc.
    llm = Ollama(model="mistral", temperature=0.0)  # temperature=0 for less hallucination
    return llm

# ---- Embedding setup ----
def local_embedding():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

# ---- Document loader ----
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

# ---- Text splitter ----
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# ---- Vector DB ----
def vector_database(chunks):
    embedding_model = local_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

# ---- Retriever ----
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

# ---- Custom prompt (system message) ----
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

# ---- QA Chain ----
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
        prompt=custom_prompt
    )
    response = qa.invoke({"query": query})
    return response['result']

# ---- Gradio UI ----
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Ollama PDF RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_application.launch(server_name="0.0.0.0", server_port=7860)
