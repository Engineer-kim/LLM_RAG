import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub



load_dotenv()

api_key = os.getenv("OPENAI_API_KEY2")
embeddings = OpenAIEmbeddings(api_key=api_key)
llm = OpenAI(api_key=api_key)

if __name__ == "__main__":
    print("hi")
    pdf_path = "/c/Users/hammo/Desktop/LLM/intro-to-vector-dbs/MyData_Introduce.pdf"
    # PDF 파일을 열고
    #
    # 각 페이지를 읽어서 텍스트를 추출하고
    #
    # 페이지 단위로 나눈 Document 객체 리스트로 변환(아래의 형태)
    # Document(
    #     page_content="This is the text of page 1 of the PDF...",
    #     metadata={'source': '/path/to/file.pdf', 'page': 0}
    # )

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    #문서를 1000자 단위로 잘라내고, 30자 겹치도록(overlap) 분할
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    # FAISS라는 벡터 DB "faiss_index_react" 이름으로 저장
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    #Select 데이터임
    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    # LLM에게 전달할 프롬프트로 구성하고, 응답을 생성
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of ReAct in 3 sentences"})
    print(res["answer"])
