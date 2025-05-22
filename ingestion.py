import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting:")
    loader = TextLoader("/c/Users/hammo/Desktop/LLM/intro-to-vector-dbs\mediumblog1.txt")
    documents = loader.load()


    print("Spliting::::")
    #원본 텍스트 문서를 최대 1000자 길이의 작은 조각들로 나누고, 이 조각들 사이에 겹치는 내용은 없도록
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting....")
    PineconeVectorStore.from_documents(texts,embeddings, index_name=os.environ("INDEX_NAME"))


    print("전과정 완료")