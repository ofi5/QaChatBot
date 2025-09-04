import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Get API keys from Streamlit secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")
google_api_key = st.secrets.get("GOOGLE_API_KEY")


st.title("QA ChatBot")

llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embeddings():
    if "vectors" not in st.session_state:
        try:
            # Initialize embeddings with proper async handling
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", 
                api_key=google_api_key,
                task_type="retrieval_document"
            )
            
            # Load and process documents
            st.session_state.loaders = PyPDFLoader("./data/sampledata.pdf")
            st.session_state.docs = st.session_state.loaders.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.chunks = st.session_state.text_splitter.split_documents(st.session_state.docs)
            st.session_state.vectors = FAISS.from_documents(st.session_state.chunks, st.session_state.embeddings)
            
        except Exception as e:
            st.error(f"Error initializing embeddings: {str(e)}")
            return False
    return True

user_question = st.text_input("Enter your question")

if st.button("Create Vector Store"):
    if vector_embeddings():
        st.write("Vector Store Created Successfully")
    else:
        st.write("Failed to Create Vector Store")
else:
    st.write("Vector Store Not Created")
    
if user_question and "vectors" in st.session_state:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_question})
        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("================================================")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
elif user_question:
    st.warning("Please create the vector store first by clicking the 'Create Vector Store' button.")

    
    
    
    
    



# st.write("""
#          This is a chatbot that can answer questions about the documents you upload.
#          """)

# pdf_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

# if pdf_files:
#     for pdf_file in pdf_files:
#         pdf_reader = PyPDFLoader(pdf_file)
#         data = pdf_reader.load()
#         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         docs = text_splitter.split_documents(data)
#         st.write(f"Loaded {len(docs)} documents from {pdf_file.name}")

#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_documents(docs, embeddings)
#     st.write("Documents loaded and embedded")

#     qa = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(model="gemini-2.0-flash"), chain_type="stuff", retriever=vectorstore.as_retriever())
#     st.write("QA chain created")
#     while True:
#         user_input = st.text_input("Enter your question")
#         if user_input:
#             response = qa.invoke({"query": user_input})
#             st.write(response["result"])
# else:
#     st.write("No PDF files uploaded")       


