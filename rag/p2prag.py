from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

def get_retriever(docs, db_path):
    # Chunk loaded documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = splitter.split_documents(docs)

    # Set embedding model
    print("Fetching embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # TODO: try other embedding models here as well
    
    print("Creating vectorstore...")
    faiss_vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    faiss_vectorstore.save_local(db_path)

    print("Creating retriever...")
    retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

    print("Retriever is ready.")
    return retriever

def get_retriever_from_db(path):
    faiss_vectorstore = FAISS.load_local(path, embeddings= OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True)
    retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
    return retriever

def make_retrieval_middleware(retriever):
    @dynamic_prompt
    def retrieval_middleware(request: ModelRequest) -> ModelRequest:
        query = request.state["messages"][-1].text
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        augmented_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
        request.prompt = augmented_prompt
        return augmented_prompt

    return retrieval_middleware

def create_rag_agent(retriever):
    print("Creating RAG agent...")
    agent = create_agent(
        model= ChatOpenAI(model="gpt-4o-mini"), # TODO: Change to standard model used for other experiments
        tools = [],
        middleware=[make_retrieval_middleware(retriever)]
    )
    return agent


