from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorDB:
    def __init__(
        self,
        vector_db_cls: Union[Chroma, FAISS] = Chroma,
        embedding=None,
    ) -> None:
        self.vector_db_cls = vector_db_cls
        self.embedding = embedding
        self.db = None

    def _build_db(self, documents):
        return self.vector_db_cls.from_documents(
            documents=documents,
            embedding=self.embedding
        )

    def get_retriever(self,
                     search_type: str = "similarity",
                     search_kwargs: dict = {"k": 10}):
        if 'k' not in search_kwargs:
            search_kwargs['k'] = 10  
        if self.db is not None:
            retriever = self.db.as_retriever(search_type=search_type,
                                             search_kwargs=search_kwargs)
        else:
            retriever = None
        return retriever
    
    def update_db(self, new_documents):
        if not new_documents:
            raise ValueError("New documents must be provided to update the database.")
        if self.db is None:
            self.db = self._build_db(new_documents)
        else:
            self.db.add_documents(new_documents)

    def clear_vectors(self):
        if self.db is None:
            raise ValueError("Database is empty, cannot clear vectors.")

        self.db = None

        self.db = self.vector_db_cls.from_documents(
            documents=[], 
            embedding=self.embedding
        )
        print("All vectors have been cleared.")