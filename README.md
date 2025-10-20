# RAG Chat Assistant for Student Loan

This is a chatbot that uses the Retrieval-Augmented Generation (RAG) architecture to answer questions about student loans in Thailand. The chatbot is built with Python and uses a local Large Language Model (LLM) to generate answers.

## Features

*   **Admin Dashboard:** A web-based dashboard for administrators to view and manage questions and answers.
*   **User Management:** A system for managing users with different roles (admin, manager, staff).
*   **RAG Architecture:** The chatbot uses a RAG architecture to retrieve relevant information from a knowledge base of PDF documents and generate accurate answers.
*   **Local LLM:** The chatbot uses a local LLM (llama3.2) to generate answers, which means that no data is sent to external services.

## How it works

The chatbot uses the following steps to answer a question:

1.  The user asks a question through the chatbot interface.
2.  The chatbot retrieves relevant information from a knowledge base of PDF documents using a sentence transformer model.
3.  The chatbot uses a local LLM to generate an answer based on the retrieved information.
4.  The chatbot displays the answer to the user.

## Technologies Used

*   **Python:** The chatbot is built with Python.
*   **Streamlit:** The admin dashboard is built with Streamlit.
*   **LangChain:** The RAG architecture is implemented with LangChain.
*   **Ollama:** The chatbot uses Ollama to run the local LLM.
*   **Sentence-Transformers:** The chatbot uses sentence-transformers to embed the documents and questions.
*   **PyMuPDF:** The chatbot uses PyMuPDF to extract text from the PDF documents.
*   **SQLite:** The chatbot uses SQLite to store questions, answers, and user information.

## Setup and Installation

1.  Clone the repository:
    ```
    git clone https://github.com/pakanan15953/RAG-Chat-Assistant-for-Student-Loan.git
    ```
2.  Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
3.  Run the chatbot:
    ```
    streamlit run chatbotv3.py
    ```
4.  Run the admin dashboard:
    ```
    streamlit run admin_dashboard.py
    ```

## Usage

To use the chatbot, simply run the `chatbotv3.py` script and open the web interface in your browser. You can then ask questions about student loans in Thailand.

To use the admin dashboard, run the `admin_dashboard.py` script and open the web interface in your browser. You can then log in with the following credentials:

*   **Username:** admin
*   **Password:** password
