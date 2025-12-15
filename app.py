from flask import Flask, request, render_template
import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ollama import Client

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Ollama LLaMA client
client = Client()

# Global variables to store vector store and retriever
faiss_store = None
retriever = None

# =============================================
# Function to extract text from PDF
# =============================================
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# =============================================
# Function to create vector store
# =============================================
def create_vector_store(text):
    global retriever
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    docs = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    global faiss_store
    faiss_store = FAISS.from_texts(docs, embeddings)
    retriever = faiss_store.as_retriever()

# =============================================
# Function to answer questions
# =============================================
def ask_question(query):
    global retriever
    # Check if retriever is initialized
    if retriever is None:
        return "Please upload a PDF file first before asking questions."
    
    # Retrieve relevant docs using LangChain's invoke method (Runnable interface)
    retrieved_docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are a helpful AI assistant. Use ONLY the following context to answer the question.
Do NOT make up information. Be concise and factual.

Context:
{context}

Question: {query}

Answer:
"""
    # Query the LLaMA model with correct API format
    response = client.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response.message.content

# =============================================
# Routes
# =============================================
@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    message = None
    if request.method == "POST":
        if "pdf" in request.files:
            pdf_file = request.files["pdf"]
            if pdf_file.filename:
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(pdf_path)
                text = extract_pdf_text(pdf_path)
                create_vector_store(text)
                message = f"PDF '{pdf_file.filename}' uploaded and processed successfully! You can now ask questions."
        elif "question" in request.form:
            question = request.form["question"]
            answer = ask_question(question)
    return render_template("index.html", answer=answer, message=message)

# =============================================
# Run the app
# =============================================
if __name__ == "__main__":
    app.run(debug=True)
