from typing import Union, List, Literal
import glob
from tqdm import tqdm
import multiprocessing
import unicodedata
from langchain_community.document_loaders import PyPDFLoader
from rag.utils import TextSplitter
import bs4
from langchain_community.document_loaders import WebBaseLoader

def remove_non_utf8_characters(text):
    try:
        return text.encode("utf-8", errors="ignore").decode("utf-8")
    except:
        return ""

def load_pdf_documents(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images=True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

def load_html_documents(links, classes = ['post-content', 'post-title', 'post-header', 'page-content']):
    web_loader = WebBaseLoader(links)

    docs = web_loader.load()

    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)


    return docs

def get_cpu_count():
    return multiprocessing.cpu_count()

class DocumentLoaderBase:
    def __init__(self) -> None:
        self.num_processes = get_cpu_count()
    
    def __call__(self, files: list[str], **kwargs):
        pass

class HTMLDocumentLoader(DocumentLoaderBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, links: list[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_link = len(links)
            with tqdm(total=total_link, desc="loading Links", unit='link') as pbar:
                if isinstance(links, str):
                    links = [links]
                for result in pool.imap_unordered(load_html_documents, links):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class PDFDocumentLoader(DocumentLoaderBase):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, files: list[str], **kwargs):
        num_processes = min(self.num_processes, kwargs["workers"])
        with multiprocessing.Pool(processes=num_processes) as pool:
            doc_loaded = []
            total_files = len(files)
            with tqdm(total=total_files, desc="loading PDFs", unit='file') as pbar:
                for result in pool.imap_unordered(load_pdf_documents, files):
                    doc_loaded.extend(result)
                    pbar.update(1)
        return doc_loaded

class DocumentLoader:
    def __init__(self,
                file_type: Literal['pdf', 'html'] = "pdf",
                split_kwargs: dict = {
                    "chunk_size": 300,
                    "chunk_overlap": 0
                }) -> None:
        
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFDocumentLoader()
        elif file_type == "html":
            self.doc_loader = HTMLDocumentLoader()
        else:
            raise ValueError("file_type must be pdf")
        
        self.doc_splitter = TextSplitter(**split_kwargs)
    
    def load(self,
            pdf_files: Union[str, List[str]],
            workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        
        doc_loaded = self.doc_loader(pdf_files , workers = workers )
        print("Loaded documents: Xin chào")
        print(doc_loaded)
        doc_split = self.doc_splitter(doc_loaded)
        return doc_split

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob(f"{dir_path}/*.pdf")
        else:
            raise ValueError("file_type must be pdf")
        
        return self.load(files, workers=workers)
