from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class ChatWithOllama:
    def __init__(self, multi_retrival=False) -> None:
        
        self.multi_retrival = multi_retrival

        self.template = """
        Your goal is to answer questions to the point in as little words as possible related to payment invoices. If for example,billing address is asked,you should give 
        address only in the form that it is originally given in the database. No need to mention anything else.
                    question: {question}

                    Consider the below context to answer the above question.
                    context: {context}
                    """

        self.chain_prompt = ChatPromptTemplate.from_template(self.template)

        self.llm = ChatOllama(model="llama3:8B")
        self.llm_multi_retriever = ChatOllama(model="mistral")

        self.embeddings = HuggingFaceEmbeddings()
        self.vector_db = FAISS.load_local("../database/pdf_db", self.embeddings, allow_dangerous_deserialization=True)

        self.retriever = self.vector_db.as_retriever()
        self.retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=self.retriever, llm=self.llm_multi_retriever
        )


        if self.multi_retrival:
            self.chain = {"context": self.retriever_from_llm , "question": RunnablePassthrough()} | self.chain_prompt | self.llm | StrOutputParser()
        else:
            self.chain = {"context": self.retriever , "question": RunnablePassthrough()} | self.chain_prompt | self.llm | StrOutputParser()

    def GetResponse(self, prompt):
        for chunk in self.chain.stream(prompt):
            if chunk == '<|eot_id|>':
                break
            yield chunk