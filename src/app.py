import os
import gradio as gr
from base.llm_model import get_llm
from rag.rag_module import Offline_RAG
from rag.file_loader import DocumentLoader
from rag.vector_store import VectorDB
from langchain_google_genai import GoogleGenerativeAIEmbeddings


os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = get_llm(
    model_name="gemini-1.5-flash-002",
    temperature=0.9,
    max_output_tokens=8192,
)

class ChatInterface:
    def __init__(self):
        self.rag_chain = None
        self.chat_chain = None
        self.vector_store = VectorDB(embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))
        self.chat_history = []

    def process_files(self, files, url_text):
        documents = []
        if files:
            doc_loader = DocumentLoader(file_type='pdf')
            pdf_paths = [file.name for file in files]
            documents.extend(doc_loader.load(pdf_paths))
        
        if url_text:
            html_loader = DocumentLoader(file_type='html')
            documents.extend(html_loader.load([url_text]))

        if documents:
            self.vector_store.update_db(documents)
            return [{'role': 'assistant', 'content': 'Documents processed successfully!'}]
        return [{'role': 'assistant', 'content': 'No documents provided.'}]

    def get_response(self, user_input, mode):
        if not user_input.strip():
            return self.chat_history

        if mode == "RAG" and self.vector_store.db is None:
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": "Please upload documents first for RAG mode."})
            return self.chat_history

        try:
            if mode == "RAG":
                chain = self.rag_chain or Offline_RAG(llm, type='rag').get_chain(self.vector_store.get_retriever())
            else:
                chain = self.chat_chain or Offline_RAG(llm, type='chat').get_chain()

            history_text = "\n".join(
                f'User: {m["content"]}' if m["role"]=="user" else f'Assistant: {m["content"]}'
                for m in self.chat_history
            )

            response = chain.invoke({
                "chat_history": history_text,
                "input": user_input
            })
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        
        return self.chat_history

    def clear_history(self):
        self.chat_history = []
        return []

    def clear_vector_store(self):
        self.vector_store.clear_vectors()
        return [{'role': 'assistant', 'content': 'Vector store cleared successfully!'}]

def main():
    chat_interface = ChatInterface()

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Document Chat Interface
            Upload documents or provide URLs to chat with their content using RAG or standard chatbot mode.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab('PDF Files'):
                    pdf_docs = gr.File(label="Upload PDF Files", file_count="multiple")
                with gr.Tab('URLs'):
                    url_text = gr.Textbox(label="Or Enter URL", placeholder="https://example.com", lines=1)
        
                mode_switch = gr.Radio(["chatbot", "RAG"], label="Select Mode", value="chatbot")
                submit_button = gr.Button("Process Documents")
                clear_vector_button = gr.Button("Clear Vector Store")
            
            with gr.Column(scale=4):
                chat_interface_md = gr.Markdown(
                    """
                    ### Instructions
                    1. Upload PDF files or enter URLs
                    2. Click 'Process Documents' to analyze the content
                    3. Select mode (RAG or chatbot)
                    4. Start chatting!
                    """
                )
                chat_history = gr.Chatbot(label="Chat History", height=400, type="messages")
                with gr.Row():
                    user_input = gr.Textbox(label="Your Question", placeholder="Ask something...", lines=2, scale=5)
                    submit_chat = gr.Button("Send", scale=1)
                clear_button = gr.Button("Clear Chat")

        submit_button.click(
            chat_interface.process_files,
            inputs=[pdf_docs, url_text],
            outputs=chat_history
        )

        user_input.submit(
            chat_interface.get_response,
            inputs=[user_input, mode_switch],
            outputs=chat_history
        )

        submit_chat.click(
            chat_interface.get_response,
            inputs=[user_input, mode_switch],
            outputs=chat_history
        )

        clear_button.click(
            chat_interface.clear_history,
            outputs=chat_history
        )

        clear_vector_button.click(
            chat_interface.clear_vector_store,
            outputs=chat_history
        )

    demo.launch()

if __name__ == "__main__":
    main()