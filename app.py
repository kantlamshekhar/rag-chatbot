import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

OPENAI_API_KEY = "Your API Key"

# Header
st.header("My Chatbot")

# Sidebar for PDF upload
with st.sidebar:
    st.title("Upload your PDF documents")
    file = st.file_uploader("Upload PDF file, and start asking questions", type="pdf")

# Question input — always visible
user_question = st.text_input("Ask a question about the PDF document")

# Only process if a PDF is uploaded
if file is not None:
    # Extract text from all pages
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Creating a vector store using FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Answer the user's question
    if user_question:
        match = vector_store.similarity_search(user_question)

        llm = ChatOpenAI(
            temperature = 0,
            max_tokens= 1000,
            model_name = "gpt-3.5-turbo"
        )

    # 3. Use the retrieved chunks to generate an answer to the user's question using a language model (e.g., GPT-3.5)
        # Build context from matched documents and call LLM directly
        context = "\n\n".join([doc.page_content for doc in match])
        prompt = f"""Use the following context to answer the question at the end.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {user_question}
Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        st.write(response.content)