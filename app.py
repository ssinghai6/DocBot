import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.llms import Ollama, OpenAI, Groq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

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

@st.cache_resource
def get_vectorstore_from_pdf(uploaded_file):
    if uploaded_file is not None:
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
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
            db = FAISS.from_documents(all_splits, embeddings)
            st.success("Document processed successfully!")
            return db
    return None

db = get_vectorstore_from_pdf(uploaded_file)

if db is not None:
    st.sidebar.title("Options")
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared successfully!")

    # Initialize the language model
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2)

    # Create the prompt template for contextualizing the question
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, db.as_retriever(), contextualize_q_prompt
    )

    # Create the prompt template for answering the question
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's question based on the below context:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


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
            
            chat_history = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.messages
            ]
            
            # Get the response from the retrieval chain
            response = retrieval_chain.invoke({"input": prompt, "chat_history": chat_history})
            answer = response['answer']

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.write("Please upload a PDF file.")

st.markdown("<p style='text-align: center; color: #888; font-size: 12px;'>Developed by <a href='https://sanshrit-singhai.vercel.app' style='color: #888; text-decoration: none;'>Sanshrit Singhai</a></p>", unsafe_allow_html=True)

