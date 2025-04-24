from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class TextSplitter:
    def __init__(self,
                seperators: List[str] = ['\n\n', '\n', ' ', ''],
                chunk_size: int = 512,
                chunk_overlap: int = 50) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            separators=seperators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def __call__(self, documents):
        return self.splitter.split_documents(documents)
