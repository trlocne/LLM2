# Document Chat Interface

This is a Gradio-based chat interface that allows users to interact with documents using RAG (Retrieval Augmented Generation) or standard chatbot mode.

## Features

- Upload and process PDF documents
- Enter URLs to process web content
- Switch between RAG and standard chatbot modes
- Interactive chat interface
- Clear chat history functionality

## Requirements

All required packages are listed in `requirements.txt`. The main dependencies include:
- gradio
- langchain
- Google Generative AI
- ChromaDB
- FAISS

```
pip install -r requirements.txt
```
## Setup
1. Clone the repository:
```bash
git clone
cd document-chat-interface
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Set up the environment variables for Google API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```
4. Run the application:
```bash
python app.py
```
5. Open your browser and go to `http://localhost:7860` to access the chat interface.
6. Upload PDF files or enter URLs to process documents.
7. Choose between RAG and standard chatbot modes.
8. Start chatting with the processed documents!

## Usage

1. Upload PDF files or enter URLs
2. Click 'Process Documents' to analyze the content
3. Select mode (RAG or chatbot)
4. Start chatting!

## Deployment

This app is ready to be deployed on Hugging Face Spaces. Make sure to set the GOOGLE_API_KEY in the Space's environment variables.