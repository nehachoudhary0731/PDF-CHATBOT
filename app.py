import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# here i Initialize session state to store chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Function to reset everything when user clicks "Reset chat"
def reset_chat():
    st.session_state.history = []
    st.session_state.pop("vector_store", None)
    st.session_state.pop("qa_chain", None)

def process_pdf(uploaded_file): # main function  to process pdf
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        # upload PDF using pyMupDF
        loader = PyMuPDFLoader(tmp_path)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)

        # Generate sentence embeddings using MiniLM model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Store in FAISS VECTOR DB
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Define prompt structure used for Q/A
        prompt_template = """Use the following context to answer the question. 
        If you don't know the answer, say you don't know. Be concise.

        Context: {context}
        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create Retrieval QA system using HuggingFace Flan-T5
        from langchain_community.llms import HuggingFaceHub
        qa_chain = RetrievalQA.from_chain_type(
    llm=HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.2, "max_length": 512},
        task="text2text-generation"
    ),
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

        # Save object to session
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain

        os.unlink(tmp_path) #Here it is clean up temp file
        return True

    except Exception as e:
        st.error(f" PDF Processing Failed: {str(e)}")
        return False

# Main Streamlit App
def main():
    st.set_page_config(page_title=" PDF Chatbot", page_icon="")
    st.title(" Chat with your PDF")
    st.caption("Upload any PDF and ask questions about its content using AI.")

    # Sidebar
    with st.sidebar:
        st.header(" Settings")
        uploaded_file = st.file_uploader("📎 Upload a PDF file", type="pdf")
        st.button(" Reset Chat", on_click=reset_chat)
        st.markdown("---")
        st.markdown("Model: Flan-T5 XXL via HuggingFace")
        st.markdown("[View Model on HuggingFace](https://huggingface.co/google/flan-t5-xxl)")

    # Process uploaded PDF
    if uploaded_file and "vector_store" not in st.session_state:
        with st.spinner("⏳ Processing PDF..."):
            if process_pdf(uploaded_file):
                st.success("✅ PDF processed successfully!")

    # Chat interface
    if "qa_chain" in st.session_state:
        # Display past chat
        for qa in st.session_state.history:
            with st.chat_message("user"):
                st.write(qa["question"])
            with st.chat_message("assistant"):
                st.write(qa["answer"])

        # New input
        query = st.chat_input("Ask a question about the PDF...")
        if query:
            st.session_state.history.append({"question": query, "answer": ""})
            with st.chat_message("user"):
                st.write(query)

            with st.spinner(" Thinking..."):
                try:
                    result = st.session_state.qa_chain({"query": query})
                    answer = result["result"]

                    # Add source pages if available
                    sources = set()
                    for doc in result["source_documents"]:
                        if "page" in doc.metadata:
                            sources.add(f"Page {doc.metadata['page']+1}")
                    if sources:
                        answer += f"\n\n*Sources:* {', '.join(sorted(sources))}"

                    st.session_state.history[-1]["answer"] = answer
                    with st.chat_message("assistant"):
                        st.write(answer)
                except Exception as e:
                    st.error(f" Error: {str(e)}")

    elif uploaded_file:
        st.warning("⚠ PDF upload failed. Try a different file.")
    else:
        st.info(" Please upload a PDF to begin.")

# Add your Hugging Face Token here
if __name__ == "__main__": 
    main()