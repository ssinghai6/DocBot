import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama, OpenAI, Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stChatInputContainer {
            background-color: #2F2F2F;
        }
        .stTextInput {
            background-color: #4F4F4F;
            color: #FFFFFF;
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    st.error("Please add your Google API key to the .env file")
    st.stop()

groq_api_key = os.getenv('groq_api_key')

if not groq_api_key:
    st.error("Please add your Groq API key to the .env file")
    st.stop()

# Streamlit app title


st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #2F2F2F; border-radius: 10px;">
        <h1 style="color: #FFFFFF; margin-bottom: 0;">DocBot</h1>
        <p style="color: #BDBDBD; margin-top: 0;">Your friendly PDF assistant</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.sidebar.title("Options")
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared successfully!")
        
    with st.spinner("Processing document..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the PDF document
        loader = PyPDFLoader('temp.pdf')
        loaded_doc = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
        all_splits = text_splitter.split_documents(loaded_doc)

        # Initialize embeddings and vector store
        # embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        db = FAISS.from_documents(all_splits, embeddings)
    st.success("Document processed successfully!")

    # Initialize the language model
    #llm = Ollama(model="llama3.2")

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

    llm = init_chat_model("google_genai:gemini-2.5-flash-lite")

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question strictly only based on the provided document.
                                              <context>
                                              {context}
                                              Question : {input}""")

    # Create the document chain and retriever
    chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, chain)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating answer..."):
            # Get the response from the retrieval chain
            response = retrieval_chain.invoke({"input": prompt})
            answer = response['answer']

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.write("Please upload a PDF file.")

st.markdown("<p style='text-align: center; color: #888; font-size: 12px;'>Developed by Sanshrit Singhai</p>", unsafe_allow_html=True)

