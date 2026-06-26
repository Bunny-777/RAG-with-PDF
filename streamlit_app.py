import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os


load_dotenv()

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 Chat with your PDF")

parser = StrOutputParser()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner=False)
def build_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatGroq(model="llama-3.3-70b-versatile")

    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    return parallel_chain | prompt | llm | parser


# --- Sidebar: upload PDF ---
with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temp path so PyPDFLoader can read it
    if "pdf_path" not in st.session_state or st.session_state.get("pdf_name") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            st.session_state["pdf_path"] = tmp.name
            st.session_state["pdf_name"] = uploaded_file.name
            st.session_state["chain"] = None  # force rebuild
            st.session_state["messages"] = []

    if st.session_state.get("chain") is None:
        with st.spinner("Reading and indexing PDF..."):
            st.session_state["chain"] = build_chain(st.session_state["pdf_path"])
        st.success(f"Ready! Ask questions about **{uploaded_file.name}**")

    # --- Chat history ---
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for role, text in st.session_state["messages"]:
        with st.chat_message(role):
            st.markdown(text)

    # --- Chat input ---
    query = st.chat_input("Ask a question about your PDF...")
    if query:
        st.session_state["messages"].append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state["chain"].invoke(query)
                st.markdown(answer)
        st.session_state["messages"].append(("assistant", answer))

else:
    st.info("👈 Upload a PDF from the sidebar to get started.")