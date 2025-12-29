import streamlit as st
import logging
import time

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
import fitz  # PyMuPDF
import base64
from langchain_core.messages import HumanMessage, AIMessage
from groq import Groq
from langchain_groq import ChatGroq

st.markdown("""
    <style>
        /* DOCBOT 2.0 - PREMIUM INTERACTIVE UI */
        
        /* Animated Gradient Background */
        .stApp {
            background: linear-gradient(-45deg, #0f0f23, #1a1a2e, #16213e, #0f2027, #1a1a2e);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            background-attachment: fixed;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Floating Orbs Effect */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: 
                radial-gradient(ellipse at 20% 80%, rgba(120, 119, 198, 0.2) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(255, 119, 198, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 40% 40%, rgba(72, 187, 255, 0.15) 0%, transparent 40%);
            pointer-events: none;
            z-index: 0;
            animation: floatOrbs 20s ease-in-out infinite;
        }
        
        @keyframes floatOrbs {
            0%, 100% { transform: translateY(0); opacity: 0.8; }
            50% { transform: translateY(-20px); opacity: 1; }
        }
        
        .stApp > header { background-color: transparent !important; }
        
        /* Premium Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(26, 26, 46, 0.95) 0%, rgba(22, 33, 62, 0.95) 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
        }
        
        section[data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }
        
        /* Glassmorphism Chat Messages with Slide Animation */
        .stChatMessage {
            background: rgba(30, 30, 46, 0.6) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            backdrop-filter: blur(20px) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
            animation: messageSlideIn 0.4s ease-out !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatMessage:hover {
            border-color: rgba(102, 126, 234, 0.3) !important;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15) !important;
            transform: translateY(-2px) !important;
        }
        
        @keyframes messageSlideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        [data-testid="stChatMessageContent"] { color: #e0e0e0 !important; }
        
        /* Premium Chat Input with Glow */
        .stChatInputContainer {
            background: rgba(30, 30, 46, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 16px !important;
            backdrop-filter: blur(20px) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        
        .stChatInputContainer:focus-within {
            border-color: rgba(102, 126, 234, 0.5) !important;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2) !important;
        }
        
        .stChatInputContainer textarea { color: #ffffff !important; }
        
        /* Animated Gradient Button */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%);
            background-size: 200% 200%;
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            background-position: 100% 0;
            transform: translateY(-3px);
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
        }
        
        /* Premium File Uploader */
        [data-testid="stFileUploader"] {
            background: rgba(30, 30, 46, 0.5);
            border-radius: 16px;
            border: 2px dashed rgba(102, 126, 234, 0.3);
            padding: 25px;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(102, 126, 234, 0.6);
            background: rgba(30, 30, 46, 0.7);
            box-shadow: 0 0 30px rgba(102, 126, 234, 0.1);
        }
        
        /* Selectbox & Toggle */
        .stSelectbox > div > div {
            background-color: rgba(30, 30, 46, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 12px !important;
            transition: all 0.3s ease !important;
        }
        
        .stSelectbox > div > div:hover { border-color: rgba(102, 126, 234, 0.4) !important; }
        .stToggle span { color: #e0e0e0 !important; }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background: rgba(30, 30, 46, 0.5) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(30, 30, 46, 0.7) !important;
            border-color: rgba(102, 126, 234, 0.3) !important;
        }
        
        /* Success/Info Messages */
        .stSuccess, .stInfo {
            background: rgba(40, 167, 69, 0.15) !important;
            border: 1px solid rgba(40, 167, 69, 0.3) !important;
            border-radius: 12px !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stInfo {
            background: rgba(102, 126, 234, 0.15) !important;
            border-color: rgba(102, 126, 234, 0.3) !important;
        }
        
        .stSpinner > div { border-top-color: #667eea !important; }
        
        /* Premium Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: rgba(30, 30, 46, 0.5); border-radius: 4px; }
        ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
        
        /* Pulse Animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-dot {
            display: inline-block;
            width: 8px; height: 8px;
            background: #28a745;
            border-radius: 50%;
            animation: pulse 2s infinite;
            margin-right: 8px;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            html, body, [class*="css"] { font-size: 16px !important; }
            h1 { font-size: 1.8rem !important; }
            section[data-testid="stSidebar"] { width: 100% !important; }
            .stButton > button, .stTextInput > div > div > input, .stSelectbox > div > div { min-height: 50px !important; font-size: 1rem !important; }
            .stChatMessage { padding: 1rem !important; margin-bottom: 0.5rem !important; border-radius: 12px !important; }
            [data-testid="stFileUploader"] { padding: 15px !important; border-radius: 12px !important; }
        }
    </style>
""", unsafe_allow_html=True)




# Load environment variables
load_dotenv()

groq_api_key = os.getenv('groq_api_key')

if not groq_api_key:
    st.error("Please add your Groq API key to the .env file")
    st.stop()

# Initialize Groq client for Vision
client = Groq(api_key=groq_api_key)

# Initialize the language model (Global init for reuse)
llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0,
        max_tokens=None,
        max_retries=2)

# Streamlit app title - Enhanced Interactive Header
st.markdown("""
    <div style="
        text-align: center; 
        padding: 30px 20px; 
        background: linear-gradient(135deg, rgba(30, 30, 46, 0.9), rgba(22, 33, 62, 0.9));
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(45deg, transparent, rgba(102, 126, 234, 0.1), transparent);
            animation: shimmer 3s ease-in-out infinite;
        "></div>
        <h1 style="
            color: #FFFFFF; 
            margin-bottom: 8px;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2, #667eea);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientText 3s ease infinite;
        ">🤖 DocBot 2.0</h1>
        <p style="color: #a0a0a0; margin-top: 0; font-size: 1.1rem;">
            Your AI-Powered PDF Assistant • Powered by <span style="color: #667eea;">Llama 3.3</span>
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px; flex-wrap: wrap;">
            <span style="
                background: rgba(102, 126, 234, 0.15);
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.85rem;
                color: #a0aec0;
                border: 1px solid rgba(102, 126, 234, 0.3);
            ">📊 Charts Analysis</span>
            <span style="
                background: rgba(118, 75, 162, 0.15);
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.85rem;
                color: #a0aec0;
                border: 1px solid rgba(118, 75, 162, 0.3);
            ">✅ Form Detection</span>
            <span style="
                background: rgba(72, 187, 120, 0.15);
                padding: 6px 14px;
                border-radius: 20px;
                font-size: 0.85rem;
                color: #a0aec0;
                border: 1px solid rgba(72, 187, 120, 0.3);
            ">🎭 Expert Modes</span>
        </div>
    </div>
    <style>
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        @keyframes gradientText {
            0%, 100% { background-position: 0% center; }
            50% { background-position: 200% center; }
        }
    </style>
""", unsafe_allow_html=True)

# Deep Visual Analysis Toggle - placed before file upload so cache key includes it
st.markdown("### 📄 Document Upload")
deep_visual_mode = st.toggle(
    "🔍 Deep Visual Analysis",
    value=False,
    help="Enable to detect tick marks, checkboxes, and form selections. Slower but more accurate for forms/surveys."
)
if deep_visual_mode:
    st.caption("*Full page analysis enabled - will detect tick marks, checkboxes, and form selections*")

# File uploader for PDF
uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)

# Reset suggestion if files change or deep visual mode changes
if uploaded_files:
    current_file_names = {f.name for f in uploaded_files}
    cache_key = (current_file_names, deep_visual_mode)
    if "last_upload_cache_key" not in st.session_state or st.session_state.last_upload_cache_key != cache_key:
        st.session_state.last_upload_cache_key = cache_key
        st.session_state.last_uploaded_files = current_file_names
        if "persona_suggestion" in st.session_state:
            del st.session_state.persona_suggestion

@st.cache_resource(show_spinner="Processing your document(s), please wait...")
def get_vectorstore_from_pdfs(uploaded_files, deep_visual_mode=False):
    if uploaded_files:
        all_splits = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Open PDF with PyMuPDF
                doc = fitz.open(temp_path)
                combined_content = []

                for page_num, page in enumerate(doc):
                    # 1. Extract Text
                    text = page.get_text()
                    
                    image_descriptions = []
                    
                    # 2. FULL PAGE ANALYSIS - Only when deep_visual_mode is enabled
                    # Captures tick marks, checkboxes, stamps, handwritten notes, etc.
                    if deep_visual_mode:
                        try:
                            # Render page at 150 DPI for good quality without huge file size
                            pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                            page_image_bytes = pix.tobytes("png")
                            encoded_page = base64.b64encode(page_image_bytes).decode('utf-8')
                            
                            max_retries = 3
                            page_description = ""
                            
                            for attempt in range(max_retries):
                                try:
                                    time.sleep(2.5)  # Rate limit throttling
                                    
                                    chat_completion = client.chat.completions.create(
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": [
                                                    {"type": "text", "text": "Analyze this entire document page. Focus on: 1) Any forms, surveys, or questionnaires - identify ALL checkboxes, radio buttons, tick marks (✓, ✔, X), filled circles, or any selection indicators. CLEARLY STATE which options are SELECTED and which are EMPTY. 2) Any charts, graphs, or diagrams - describe data trends and key values. 3) Any tables - describe the structure and key data. 4) Any handwritten notes, stamps, or signatures. Be thorough and explicit about selection states."},
                                                    {
                                                        "type": "image_url",
                                                        "image_url": {
                                                            "url": f"data:image/png;base64,{encoded_page}",
                                                        },
                                                    },
                                                ],
                                            }
                                        ],
                                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                                    )
                                    page_description = chat_completion.choices[0].message.content
                                    break
                                except Exception as e:
                                    if "429" in str(e) and attempt < max_retries - 1:
                                        st.toast(f"Rate limit hit on page {page_num+1}, retrying...", icon="⏳")
                                        time.sleep(5)
                                        continue
                                    else:
                                        raise e
                            
                            if page_description:
                                image_descriptions.append(f"[FULL PAGE VISUAL ANALYSIS (Page {page_num+1})]: {page_description}")
                                
                        except Exception as e:
                            print(f"Error in full page analysis for page {page_num+1}: {e}")
                    
                    # 3. Extract Embedded Images (charts, figures embedded separately)
                    images = page.get_images(full=True)
                    
                    for img_index, img in enumerate(images):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Analyze image with Vision Model
                        try:
                            # Encode image to base64
                            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                            
                            # Retry logic for Rate Limits (429)
                            max_retries = 3
                            description = ""
                            
                            for attempt in range(max_retries):
                                try:
                                    # Rate limit throttling: Sleep to respect 30 RPM (1 req / 2s). 
                                    # Sleeping 2.5s is safe.
                                    time.sleep(2.5)
                                    
                                    chat_completion = client.chat.completions.create(
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": [
                                                    {"type": "text", "text": "Analyze this image in detail. If it's a chart or graph, describe the data trends, axes, and key values. If it's a diagram, explain the flow. If text, transcribe it. IMPORTANT: If this is a form, survey, or questionnaire, carefully identify ALL checkboxes, radio buttons, tick marks, and selection indicators. Clearly state which options are SELECTED (marked with ✓, ✔, X, filled circles, highlights, or any other selection indicator) and which are EMPTY/UNSELECTED. List all options and their selection status explicitly."},
                                                    {
                                                        "type": "image_url",
                                                        "image_url": {
                                                            "url": f"data:image/jpeg;base64,{encoded_image}",
                                                        },
                                                    },
                                                ],
                                            }
                                        ],
                                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                                    )
                                    description = chat_completion.choices[0].message.content
                                    break # Success, exit retry loop
                                except Exception as e:
                                    if "429" in str(e) and attempt < max_retries - 1:
                                        # Rate limit hit, wait longer
                                        st.toast(f"Rate limit hit, retrying image {img_index+1}...", icon="⏳")
                                        time.sleep(5)
                                        continue
                                    else:
                                        raise e # Other error or max retries reached

                            if description:
                                image_descriptions.append(f"[IMAGE DESCRIPTION (Page {page_num+1}, Image {img_index+1})]: {description}")
                                
                        except Exception as e:
                            print(f"Error processing image: {e}")
                            continue

                    # Combine text and image descriptions
                    page_content = text + "\n\n" + "\n\n".join(image_descriptions)
                    
                    # Create a Document object directly (simulating what PyPDFLoader does but better)
                    from langchain_core.documents import Document
                    combined_content.append(Document(page_content=page_content, metadata={"source": temp_path, "page": page_num}))

                # Split the combined document content
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100) # Increased chunk size for descriptions
                splits = text_splitter.split_documents(combined_content)
                all_splits.extend(splits)
                
                doc.close()

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Initialize embeddings and vector store
        if all_splits:
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
            # Using ChromaDB might be better for larger contexts, but sticking to FAISS as per original
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
        Prompt:
        Analyze the following document excerpt (which may include text and descriptions of visual content like charts/graphs) and determine the SINGLE best Expert Persona to answer questions about it.
        
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

db = get_vectorstore_from_pdfs(uploaded_files, deep_visual_mode)

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
    system_prompt_with_formatting = base_system_prompt + """

ABSOLUTELY CRITICAL - RESPONSE FORMATTING:
You MUST follow these rules STRICTLY:

1. FORBIDDEN - NEVER OUTPUT THESE PATTERNS:
   - NO: $4,090.30+$2,405.33 = $6,495.63
   - NO: 4,090.30+2,405.33+4,858.28=...
   - NO: Any inline math expressions with + signs between numbers
   - NO: LaTeX notation of any kind

2. REQUIRED FORMAT FOR CALCULATIONS:
   Use tables or bullet lists:
   | Item | Amount |
   |------|--------|
   | Earnings | $4,090.30 |
   | Holiday | $2,405.33 |
   | **Total** | **$6,495.63** |

3. FIX ALL TEXT EXTRACTION ERRORS:
   - Fix spaces inside numbers: "30, 000" → "30,000", "25, 000" → "25,000"
   - Fix merged words: "witha" → "with a", "withthe" → "with the"
   - Fix missing spaces: "250deductible" → "$250 deductible"
   - Proper currency format: Always use $X,XXX.XX format
   
4. NEVER chain numbers with + signs in a single line.
5. Always present data in clean, readable format."""
    
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

