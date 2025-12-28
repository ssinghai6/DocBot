import streamlit as st
import logging

# Set logging level for libraries that might be too verbose
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

st.markdown("""
    <style>
        /* Modern gradient background */
        .stApp {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #1a1a2e 75%, #0f0f23 100%);
            background-attachment: fixed;
        }
        
        /* Subtle animated gradient overlay */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(ellipse at 20% 80%, rgba(120, 119, 198, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at 40% 40%, rgba(72, 187, 255, 0.1) 0%, transparent 40%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* Main content styling */
        .stApp > header {
            background-color: transparent !important;
        }
        
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        section[data-testid="stSidebar"] .stMarkdown {
            color: #e0e0e0;
        }
        
        /* Chat container styling */
        .stChatMessage {
            background-color: rgba(30, 30, 46, 0.7) !important;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* User message styling */
        [data-testid="stChatMessageContent"] {
            color: #e0e0e0 !important;
        }
        
        /* Chat input styling */
        .stChatInputContainer {
            background-color: rgba(30, 30, 46, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(10px);
        }
        
        .stChatInputContainer textarea {
            color: #ffffff !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background-color: rgba(30, 30, 46, 0.6);
            border-radius: 12px;
            border: 2px dashed rgba(255, 255, 255, 0.2);
            padding: 20px;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(102, 126, 234, 0.5);
        }
        
        /* Selectbox styling */
        .stSelectbox > div > div {
            background-color: rgba(30, 30, 46, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 8px;
        }
        
        /* Toggle styling */
        .stToggle span {
            color: #e0e0e0 !important;
        }
        
        /* Success/error message styling */
        .stSuccess {
            background-color: rgba(40, 167, 69, 0.2) !important;
            border: 1px solid rgba(40, 167, 69, 0.4) !important;
        }
        
        /* Spinner styling */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(30, 30, 46, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
    </style>
""", unsafe_allow_html=True)


# Load environment variables
load_dotenv()

groq_api_key = os.getenv('groq_api_key')

if not groq_api_key:
    st.error("Please add your Groq API key to the .env file")
    st.stop()

# Initialize the language model (Global init for reuse)
llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0,
        max_tokens=None,
        max_retries=2)

# Streamlit app title


st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #2F2F2F; border-radius: 10px;">
        <h1 style="color: #FFFFFF; margin-bottom: 0;">DocBot</h1>
        <p style="color: #BDBDBD; margin-top: 0;">Your friendly PDF assistant</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for PDF
# File uploader for PDF
uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

# Reset suggestion if files change
if uploaded_files:
    current_file_names = {f.name for f in uploaded_files}
    if "last_uploaded_files" not in st.session_state or st.session_state.last_uploaded_files != current_file_names:
        st.session_state.last_uploaded_files = current_file_names
        if "persona_suggestion" in st.session_state:
            del st.session_state.persona_suggestion

@st.cache_resource(show_spinner="Processing your document(s), please wait...")
def get_vectorstore_from_pdfs(uploaded_files):
    if uploaded_files:
        all_splits = []
        for uploaded_file in uploaded_files:
            # Use original filename to prevent collisions effectively, or just a safe temp name
            # Since we process sequentially inside the loop, we can reuse a temp name, 
            # OR better, use the file's name to be safe if parallelization ever happens (not here though).
            # Simple approach: write to a temp file, load, then delete.
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Load the PDF document
                loader = PyPDFLoader(temp_path)
                loaded_doc = loader.load()

                # Split the document into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
                splits = text_splitter.split_documents(loaded_doc)
                all_splits.extend(splits)
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Initialize embeddings and vector store
        if all_splits:
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
            db = FAISS.from_documents(all_splits, embeddings)
            return db
    return None

def analyze_document_persona(db, llm):
    """Analyzes the document to suggest the best expert persona."""
    try:
        # Get a glimpse of the document content
        retriever = db.as_retriever(search_kwargs={"k": 3})
        # General query to get the gist
        docs = retriever.invoke("Overview of the document content and purpose")
        context_text = "\n\n".join([d.page_content for d in docs])[:3000]

        prompt = f"""
        Analyze the following document excerpt and determine the SINGLE best Expert Persona to answer questions about it.
        
        Options:
        - Doctor
        - Finance Expert
        - Engineer
        - AI/ML Expert
        - Lawyer
        - Consultant
        - Generalist

        Document Excerpt:
        {context_text}

        Return ONLY the exact name of the persona.
        """
        response = llm.invoke(prompt)
        suggestion = response.content.strip()
        
        # Clean up response (sometimes LLMs add extra chars)
        for persona in EXPERT_PERSONAS.keys():
            if persona in suggestion:
                return persona
        return "Generalist"
    except Exception as e:
        print(f"Error analyzing persona: {e}")
        return "Generalist"

db = get_vectorstore_from_pdfs(uploaded_files)

# Define expert personas with their system prompts
EXPERT_PERSONAS = {
    "Generalist": {
        "icon": "🎯",
        "description": "Balanced, general-purpose assistant",
        "system_prompt": "You are a knowledgeable assistant. Answer the question strictly based ONLY on the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.' Do not use outside knowledge. Maintain a helpful and clear tone."
    },
    "Doctor": {
        "icon": "🩺",
        "description": "Medical & healthcare perspective",
        "system_prompt": "You are a Medical Doctor with extensive clinical experience. Answer the question strictly based ONLY on the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.' Do not use outside knowledge. Approach the answer from a medical/healthcare perspective. Use appropriate medical terminology when relevant, explain health implications clearly, and always emphasize the importance of consulting healthcare professionals for medical decisions. Maintain a professional, empathetic, and cautious tone."
    },
    "Finance Expert": {
        "icon": "💰",
        "description": "Financial & investment analysis",
        "system_prompt": "You are a Senior Finance Expert with deep knowledge in investment, banking, and financial analysis. Answer the question strictly based ONLY on the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.' Do not use outside knowledge. Analyze information from a financial perspective, consider risk factors, ROI implications, and market dynamics where relevant. Use financial terminology appropriately and provide insights that would be valuable for financial decision-making. Maintain an analytical, precise, and professional tone."
    },
    "Engineer": {
        "icon": "⚙️",
        "description": "Technical & engineering focus",
        "system_prompt": "You are a Senior Engineer with expertise in systems design, problem-solving, and technical implementation. Answer the question strictly based ONLY on the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.' Do not use outside knowledge. Approach problems systematically, consider technical specifications, feasibility, and implementation details. Break down complex concepts into understandable components. Focus on practical solutions and engineering best practices. Maintain a logical, methodical, and solution-oriented tone."
    },
    "AI/ML Expert": {
        "icon": "🤖",
        "description": "AI, ML & data science insights",
        "system_prompt": "You are an AI/ML Expert with deep knowledge in machine learning, deep learning, data science, and artificial intelligence systems. Answer the question strictly based ONLY on the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.' Do not use outside knowledge. Analyze information through the lens of data patterns, algorithmic approaches, and AI/ML methodologies. Discuss model architectures, training considerations, and evaluation metrics where relevant. Maintain a technical yet accessible tone, making complex AI concepts understandable."
    },
    "Lawyer": {
        "icon": "⚖️",
        "description": "Legal analysis & compliance",
        "system_prompt": "You are a Senior Lawyer with expertise in legal analysis, contracts, and regulatory compliance. Answer the question strictly based ONLY on the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.' Do not use outside knowledge. Analyze information from a legal perspective, identify potential legal implications, contractual obligations, and compliance considerations. Use precise legal language when appropriate but ensure explanations are accessible. Always note that this is not formal legal advice. Maintain a careful, thorough, and authoritative tone."
    },
    "Consultant": {
        "icon": "📊",
        "description": "Strategic business advisory",
        "system_prompt": "You are a Senior Consultant with extensive experience in strategy, operations, and business transformation. Answer the question strictly based ONLY on the following context:\n\n{context}\n\nIf the answer is not in the context, say 'I cannot answer this based on the provided document.' Do not use outside knowledge. Provide strategic insights, consider business implications, stakeholder impacts, and actionable recommendations. Structure your responses clearly with key takeaways. Maintain a professional, authoritative, and solution-focused tone."
    }
}

if db is not None:
    # --- Auto-Analysis for Persona Suggestion ---
    # Check if we have a suggestion for the current vectorstore
    if "persona_suggestion" not in st.session_state:
        with st.spinner("🤖 Analyzing document to suggest expert mode..."):
            suggestion = analyze_document_persona(db, llm)
            st.session_state.persona_suggestion = suggestion
            st.toast(f"Suggested Mode: {suggestion}", icon=EXPERT_PERSONAS[suggestion]['icon'])

    st.sidebar.title("Options")
    
    # Expert Persona Selector
    st.sidebar.markdown("### 🎭 Expert Mode")
    
    # Show suggestion alert
    suggested = st.session_state.persona_suggestion
    if suggested != "Generalist":
        st.sidebar.info(f"💡 based on your document, **{suggested}** mode is recommended.")

    selected_persona = st.sidebar.selectbox(
        "Select response style:",
        options=list(EXPERT_PERSONAS.keys()),
        index=list(EXPERT_PERSONAS.keys()).index(suggested),  # Default to suggested
        format_func=lambda x: f"{EXPERT_PERSONAS[x]['icon']} {x}"
    )
    st.sidebar.caption(f"*{EXPERT_PERSONAS[selected_persona]['description']}*")
    
    # Store selected persona in session state
    st.session_state.selected_persona = selected_persona
    
    # Deep Research Mode toggle (only for non-Generalist personas)
    if selected_persona != "Generalist":
        st.sidebar.markdown("### 🔬 Analysis Mode")
        deep_research = st.sidebar.toggle(
            "Deep Research",
            value=st.session_state.get('deep_research', False),
            help="Enable extremely deep, multi-angle analysis with step-by-step reasoning"
        )
        st.session_state.deep_research = deep_research
        if deep_research:
            st.sidebar.caption("*🧠 Deep reasoning enabled - responses will be more thorough*")
    else:
        st.session_state.deep_research = False
    
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat cleared successfully!")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Contribute")
    st.sidebar.markdown("Enjoying the app? Please consider supporting its development.")
    st.sidebar.markdown(
        """
        <a href="https://www.paypal.com/donate/?business=G5C3WRTY7YTXC&no_recurring=0&currency_code=USD" target="_blank">
            <img src="https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif" alt="Donate with PayPal" />
        </a>
        """,
        unsafe_allow_html=True,
    )



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

    # Get the system prompt based on selected persona
    persona = st.session_state.get('selected_persona', 'Generalist')
    deep_research = st.session_state.get('deep_research', False)
    base_system_prompt = EXPERT_PERSONAS[persona]['system_prompt']
    
    # Add deep research instructions if enabled
    if deep_research and persona != "Generalist":
        deep_research_addon = """

🔬 DEEP RESEARCH MODE ACTIVATED:
You must provide an EXTREMELY thorough and rigorous analysis. Follow this structured approach:

1. **Initial Assessment**: Start by clearly stating your understanding of the question and its scope.

2. **Multi-Angle Analysis**: Examine the topic from multiple perspectives relevant to your expertise:
   - Consider different scenarios and edge cases
   - Identify assumptions and their implications
   - Explore both obvious and non-obvious aspects

3. **Evidence-Based Reasoning**: For each point:
   - Quote or reference specific parts of the document
   - Explain your reasoning chain step-by-step
   - Acknowledge any uncertainties or limitations in the data

4. **Critical Evaluation**:
   - Identify potential gaps or inconsistencies in the information
   - Consider what additional information would strengthen the analysis
   - Weigh pros and cons where applicable

5. **Synthesis & Conclusions**:
   - Summarize key findings with confidence levels
   - Provide actionable insights or recommendations
   - Highlight any caveats or areas requiring further investigation

Take your time to think deeply. Quality and thoroughness are more important than brevity."""
        base_system_prompt = base_system_prompt + deep_research_addon
    
    # Add formatting correction instruction to all prompts
    system_prompt_with_formatting = base_system_prompt + "\n\nIMPORTANT: The context may contain formatting errors like missing spaces between numbers (e.g., '100to200'). You MUST correct these in your response (e.g., write '100 to 200')."
    
    # Create the prompt template for answering the question
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_with_formatting),
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
    st.write("Please upload PDF file(s).")

st.markdown("<p style='text-align: center; color: #888; font-size: 12px;'>Developed by <a href='https://sanshrit-singhai.vercel.app' style='color: #00C4FF; text-decoration: none;'>Sanshrit Singhai</a></p>", unsafe_allow_html=True)

