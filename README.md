# 🤖 DocBot 2.0: Advanced Smart PDF Assistant

**DocBot** is a next-generation conversational AI assistant that turns your static PDF documents into interactive, intelligent conversations. It uses advanced RAG (Retrieval-Augmented Generation) to answer questions strictly based on your data, with zero hallucinations.

Now supercharged with **Expert Personas**, **Deep Research capabilities**, and powered by **Meta's Llama 3.3 70B** via Groq for uncompromised speed and reasoning.

We now also support **running completely locally** with Ollama!

---

## 🚀 Key Features

### 🎭 Smart Expert Personas
DocBot doesn't just answer; it adopts the mindset of an expert tailored to your needs:
- **🩺 Doctor**: Medical analysis with empathy & clinical precision.
- **💰 Finance Expert**: ROI, risk assessment, and market analysis.
- **⚙️ Engineer**: Technical specifications, system design, and feasibility.
- **⚖️ Lawyer**: Contract analysis, compliance, and regulatory insights.
- **🤖 AI/ML Expert**: Data science, algorithms, and model architecture.
- **📊 Consultant**: Strategic business advice and actionable insights.
- **🎯 Generalist**: Balanced, clear, and helpful for any topic.

### 👁️ Multimodal Vision Analysis
DocBot is now fully multimodal! It **sees and understands** images, charts, and graphs within your PDFs.
*   **Extracts Images**: Automatically pulls text and visuals from every page.
*   **Vision AI**: Uses **Llama 4 Scout (17B)** (Cloud) or **Llama 3.2 Vision** (Local) to describe charts, graphs, and diagrams in detail.
*   **Integrated Context**: Image descriptions are indexed so you can ask questions like *"What is the trend in the sales graph on page 3?"*
*   **Form & Checkbox Detection**: Intelligently identifies checkboxes, radio buttons, tick marks (✓, ✔, X), and selection indicators in forms, surveys, and questionnaires—accurately reporting which options are selected vs. empty.

### 🧠 Auto-Magical Suggestion
Upload a PDF, and DocBot **automatically reads and analyzes it** to recommend the perfect expert mode for you.
> *Upload a medical report? DocBot suggests "Doctor Mode".*
> *Upload a balance sheet? DocBot suggests "Finance Expert Mode".*

### 🔬 Deep Research Mode
Need more than a quick answer? Toggle **Deep Research** (available in expert modes) to activate:
- **Multi-Angle Analysis**: Examines edge cases and assumptions.
- **Step-by-Step Reasoning**: Logical breakdowns of complex topics.
- **Evidence-Based Answers**: Strict citations from your documents.

### ⚡ Unlimited Free Intelligence
- **Powered by Llama 3.3 70B**: One of the world's most advanced open-source models.
- **Blazing Fast**: Hosted on Groq LPU™ Inference Engine for instant responses.
- **Free**: No paid API keys required for standard usage.

### 🎨 Modern "Deep Space" UI
- **Glassmorphism Design**: Sleek, translucent cards and blurred backgrounds.
- **Dynamic Gradients**: A stunning deep space theme that looks professional.
- **Smooth Animations**: Interactive elements that feel alive.

---

## 🛠️ Getting Started (Cloud / Groq)

This is the easiest way to start with no hardware requirements.

### Prerequisites
- Python 3.8+
- [Groq API Key](https://console.groq.com/) (Free)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd DocBot
    ```

2.  **Create a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the root folder:
    ```env
    # Only Groq is needed now!
    GROQ_API_KEY="your_groq_api_key_here"
    ```

5.  **Run the App:**
    ```sh
    streamlit run app.py
    ```

---

## 🔒 Getting Started (Local / Ollama)

Run everything 100% locally on your machine for maximum privacy.

### Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- A machine with decent RAM (8GB+ recommended, 16GB+ for best results).

### Setup

1.  **Pull Recommended Models**:
    Run these commands in your terminal:
    ```sh
    # Chat & Embeddings (Lightweight)
    ollama pull llama3.2

    # Advanced Reasoning (Optional, requires more RAM)
    ollama pull deepseek-r1

    # Vision Model (Required for image analysis)
    ollama pull llava  # Standard
    # OR
    ollama pull llama3.2-vision # Advanced (Requires update in sidebar)
    ```

2.  **Run the Local App**:
    ```sh
    streamlit run app_ollama.py
    ```

3.  **Local Configuration**:
    - Open the **Sidebar**.
    - Enter the names of the models you pulled (e.g., set Chat Model to `deepseek-r1` or `llama3.2`).
    - Everything runs locally!

---

## 💡 How to Use

1.  **Upload**: Drag & drop your PDF(s) into the sidebar.
2.  **Wait for Suggestion**: Watch for the "💡 Suggested Mode" alert.
3.  **Select Mode**: Confirm the suggested expert or pick your own from the dropdown.
4.  **Deep Research**: (Optional) Toggle "Deep Research" for complex queries.
5.  **Chat**: Ask questions and get expert-level, cited answers!

---

**Built with:** [Streamlit](https://streamlit.io/) • [LangChain](https://www.langchain.com/) • [Groq](https://groq.com/) • [Ollama](https://ollama.com/) • [FAISS](https://github.com/facebookresearch/faiss)
