from flask import Flask, request, jsonify
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DocumentLoader

#from langchain.document_loaders import SimpleDocumentLoader
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import PyPDF2

# Initialize Flask app
app = Flask(__name__)

# Initialize Gemini model
try:
    client = genai.Client(api_key='AIzaSyA3n-icZ6sukOHGIEiGeYkvV9wp8EFHr8o')
    model = client.models.get_model("gemini-pro")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")

# Load PDF and split into chunks
def load_pdf_and_split(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks

# Generate embeddings using Gemini model
def generate_embeddings(chunks):
    embeddings = []
    try:
        for chunk in chunks:
            response = model.embeddings.create(text=chunk)
            embeddings.append(response['embedding'])
    except Exception as e:
        print(f"Error generating embedding: {e}")
    return embeddings

# Create FAISS index and add embeddings
def create_faiss_index(embeddings):
    try:
        faiss_index = FAISS()
        faiss_index.add_embeddings(embeddings)
        return faiss_index
    except Exception as e:
        print(f"Error creating FAISS index: {e}")

# Create a retrieval QA chain
def create_retrieval_qa_chain(faiss_index):
    try:
        embeddings = OpenAIEmbeddings()
        retriever = faiss_index.as_retriever(embeddings)
        chain = RetrievalQA.from_chain_type(
            retriever=retriever,
            chain_type="stuff", 
            chain_kwargs={"qa_chain": load_qa_chain(qa_type="extractive")}
        )
        return chain
    except Exception as e:
        print(f"Error creating retrieval QA chain: {e}")

# Routes
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        question = request.json.get('question')
        faiss_index = create_faiss_index(generate_embeddings(load_pdf_and_split('your_pdf_path.pdf')))
        chain = create_retrieval_qa_chain(faiss_index)
        answer = chain.run(question)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /ask route: {e}")
        return jsonify({'error': 'An error occurred'}), 500

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Flask app with Gemini API integration."

if __name__ == '__main__':
    app.run(debug=True)
