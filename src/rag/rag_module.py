import re
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer\s*:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text.split("\n")[0].strip()
        else:
            return text_response.strip()


class Offline_RAG:
    def __init__(self, llm, type: str = 'rag') -> None:
        self.typ = type
        self.llm = llm
        self.str_parser = Str_OutputParser()

        if self.typ == 'rag':
            self.prompt = self._prompt_rag_context()
        else:
            self.prompt = self._prompt_context()

    def get_chain(self, retriever=None):

        def combine_docs(input_dict):
            docs = retriever.get_relevant_documents(input_dict["input"])
            input_dict["context"] = self._format_docs(docs)
            return input_dict

        chain = RunnablePassthrough.assign(
            chat_history=lambda d: d["chat_history"],
            input=lambda d: d["input"]
        )

        if self.typ == 'rag':
            chain = chain | RunnableLambda(combine_docs)

        chain = chain | self.prompt | self.llm | RunnableLambda(lambda x: self.str_parser.parse(
            x if isinstance(x, str) else getattr(x, "content", str(x))
        ))
        return chain

    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def _prompt_rag_context(self):
        tpl = """
            Bạn là trợ lý AI, sử dụng phần Context dưới đây để trả lời. Nếu không biết, hãy thành thật nói không biết. Trả lời ngắn gọn trong tối đa ba câu.

            Hội thoại trước đó:
            {chat_history}

            User: {input}
            Context: {context}
            Answer:
        """
        return PromptTemplate(
            input_variables=["chat_history", "input", "context"],
            template=tpl
        )

    def _prompt_context(self):
        tpl = """
            Đây là cuộc trò chuyện giữa người và trợ lý AI. AI hoạt bát, cung cấp chi tiết, nếu không biết thì nói không biết.

            Hội thoại:
            {chat_history}

            User: {input}
            Answer:
        """
        return PromptTemplate(
            input_variables=["chat_history", "input"],
            template=tpl
        )
