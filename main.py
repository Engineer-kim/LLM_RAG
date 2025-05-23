import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from openai import vector_stores

load_dotenv()


if __name__ == '__main__':
    print("Retriving....")

    embeddings = OpenAIEmbeddings() #텍스트를 숫자 벡터로 변환 => 임베딩
    llm = ChatOpenAI()

    query = "what is Pinecone in machin learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"),
        embedding=embeddings,
    )

    #대규모 언어 모델이 검색된 정보와 대화 이력을 활용하여 사용자 질문에 정확하게 답변하도록 안내하는 표준화된 프롬프트 템플릿을 LangChain Hub에서 가져옴
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    # "LLM에게 답변을 만들기 위한 정보를 잘 정리해주는" 역할(상상이 아닌 실제로 존재하는 자료 기반)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)


    #바로 앞줄에서 찿은 벡터 DB에서 정보를 찾고(검색), 찾은 정보를 종합하여 LLM이 실제 답변을 생성
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})

    print(result)