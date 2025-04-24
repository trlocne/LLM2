import os
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "AIzaSyA2x2QwSCbuTYoEP8VROnjRtz7-d3_3uik"

def get_llm(
    model_name: str = "gemini-1.5-flash-002", 
    temperature: float = 0.9,
    max_output_tokens: int = 8192,
    **kwargs
):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs
    )


    return llm
