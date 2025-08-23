import gradio as gr
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


import warnings
warnings.filterwarnings('ignore')

# List your available models here
LLM_MODELS = ["mistral", "llama2", "phi3"]
EMBEDDING_MODELS = [
    r"C:\models\all-MiniLM-L6-v2",
    r"C:\models\paraphrase-MiniLM-L3-v2"
]

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
    return Ollama(model=llm_model_name, temperature=0.0)

def local_embedding(embedding_model_name):
    return HuggingFaceEmbeddings(model_name=embedding_model_name)

def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def vector_database(chunks, embedding_model_name):
    embedding_model = local_embedding(embedding_model_name)
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

def retriever(file, embedding_model_name):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks, embedding_model_name)
    retriever = vectordb.as_retriever()
    return retriever

def retriever_qa(file, query, llm_model_name, embedding_model_name):
    llm = get_llm(llm_model_name)
    retriever_obj = retriever(file, embedding_model_name)

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
    response = qa({"query": query})
    answer = response['result']
    sources = response.get('source_documents', [])
    source_texts = "\n\n---\n\n".join(
        [f"Source {i+1}:\n{doc.page_content[:1000]}" for i, doc in enumerate(sources[:3])]
    ) if sources else "No sources found."
    return answer, source_texts


rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here..."),
        gr.Dropdown(label="Choose LLM Model", choices=LLM_MODELS, value=LLM_MODELS[0]),
        gr.Dropdown(label="Choose Embedding Model", choices=EMBEDDING_MODELS, value=EMBEDDING_MODELS[0])
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Source Chunks Used")
    ],
    title="Ollama PDF RAG Chatbot",
    description="Upload a PDF, select models, and ask any question. The chatbot will answer using the document and show the source text."
)

rag_application.launch(server_name="0.0.0.0", server_port=7860)
