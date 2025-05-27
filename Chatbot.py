import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import os

# 🔐 Load OpenAI API key from Streamlit secrets or environment variable
api_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 🌟 UI Setup
st.header("📄 PDF ChatBot using OpenAI GPT")

# 📥 PDF Upload
with st.sidebar:
    st.title("Upload Your PDF")
    file = st.file_uploader("Choose a PDF file", type="pdf")

# 📃 PDF Processing
if file is not None:
    text = ""
    table_texts = []

    def clean_row(row):
        return [cell if cell is not None else "" for cell in row]

    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

            tables = page.extract_tables()
            for table in tables:
                if table:
                    header = clean_row(table[0])
                    markdown_table = "| " + " | ".join(header) + " |\n"
                    markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
                    for row in table[1:]:
                        cleaned_row = clean_row(row)
                        markdown_table += "| " + " | ".join(cleaned_row) + " |\n"
                    table_texts.append(markdown_table)

    # Combine extracted text and tables
    text += "\n\n" + "\n\n".join(table_texts)

    # 🧩 Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=2000,
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # 🔍 Vector embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    vector_store = FAISS.from_texts(chunks, embeddings)

    # 🧠 User question
    user_question = st.text_input("❓ Ask a question about the PDF")

    if user_question:
        # Find relevant chunks
        docs = vector_store.similarity_search(user_question, k=6)
        context = "\n\n".join([doc.page_content for doc in docs])

        # 👇 Prompt with instructions
        prompt = f"""You are an assistant. Use the context below to answer the question.
If the answer includes a table, respond using proper Markdown format.
If not found in the context, reply "I don't know".

Context:
{context}

Question: {user_question}
Answer:"""

        st.markdown("🧠 Thinking with OpenAI GPT...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=512,
                temperature=0.3
            )
            answer = response.choices[0].message.content
            st.write("🧠 Answer:")
            if "|" in answer:
                st.markdown(answer)
            else:
                st.success(answer)
        except Exception as e:
            st.error(f"❌ Error from OpenAI API: {e}")
