# DocBot: Your Conversational PDF Assistant

DocBot is an intelligent, conversational PDF assistant built with Streamlit and LangChain. Upload any PDF document and ask questions in a natural, conversational manner. The chatbot remembers the context of your conversation to provide accurate, relevant answers based on the document's content.

## Key Features

- **Interactive Chat Interface**: Ask questions about your PDF documents through a user-friendly chat UI.
- **Conversational Memory**: The chatbot maintains a "chain of thought," allowing you to ask follow-up questions that reference the previous context.
- **Efficient Document Processing**: Utilizes caching to process each PDF only once, ensuring fast and responsive interactions for subsequent questions.
- **Powered by LangChain & Gemini**: Leverages the power of LangChain for retrieval-augmented generation (RAG) and Google's Gemini Pro for intelligent, context-aware responses.

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

- Python 3.8+
- A virtual environment tool (like `venv`)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Configure your API keys:**
    Create a `.env` file in the root of the project and add your API keys:
    ```env
    GOOGLE_API_KEY="your_google_api_key"
    GROQ_API_KEY="your_groq_api_key"
    ```

5.  **Run the Streamlit application:**
    ```sh
    streamlit run app.py
    ```

The application will now be running and accessible in your web browser.
