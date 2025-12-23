from fastapi import FastAPI, UploadFile, File, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import shutil
import tempfile
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity in dev/demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}]
    document_text: str
    question: str

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Handle PDF upload and return extracted text.
    Targeting stateless Vercel deployment, so we return text to client.
    """
    try:
        print(f"Received upload request: {file.filename}")
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name

        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        
        # Combine all text
        full_text = "\n\n".join([page.page_content for page in pages])
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return {"text": full_text, "filename": file.filename}
    except Exception as e:
        return Response(content=f"Error processing PDF: {str(e)}", status_code=500)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint using Gemini 1.5 Flash with long context.
    Context is passed in the request body from the client.
    """
    if "GOOGLE_API_KEY" not in os.environ:
         return Response(content="GOOGLE_API_KEY not found in environment", status_code=500)

    try:
        # Initialize Gemini 1.5 Flash
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            temperature=0.3,
            max_retries=2
        )

        # Create prompt with history and context
        # We simplify history handling by appending it to the context or using a specific prompt structure
        
        system_prompt = """You are DocBot, a helpful assistant. 
        Use the following document text to answer the user's question. 
        If the answer is not in the document, say so.
        
        Document Content:
        {doc_text}
        """
        
        messages = [
            ("system", system_prompt.format(doc_text=request.document_text)),
        ]
        
        # Append chat history (last 10 messages to avoid getting too huge, though 1.5 Flash is fine with it)
        for msg in request.messages:
             messages.append((msg["role"], msg["content"]))
             
        # Append latest question if not already in messages (depending on frontend impl)
        # Assuming frontend sends history WITHOUT the current new question, or WITH it. 
        # Let's assume frontend sends `messages` as history, and `question` as the new trigger.
        messages.append(("human", request.question))

        # Invoke LLM
        response = llm.invoke(messages)
        
        return {"answer": response.content}

    except Exception as e:
        return Response(content=f"Error generating response: {str(e)}", status_code=500)
