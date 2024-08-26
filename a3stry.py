from flask import Flask, request, Response
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import faiss
import pdfplumber

# Flask application setup
app = Flask(__name__)

# Set your API key for Gemini
os.environ['GOOGLE_API_KEY'] = "AIzaSyA3n-icZ6sukOHGIEiGeYkvV9wp8EFHr8o"

# Configure the API for Gemini
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')

# Assuming Gemini provides an embeddings API, initialize it
class GeminiEmbeddings:
    def __init__(self):
        pass

    def embed_documents(self, documents):
        embedded_docs = []
        for doc in documents:
            embedding = self.generate_embedding(doc)
            embedded_docs.append(embedding)
        return embedded_docs

    def generate_embedding(self, document):
        # Dummy embedding; replace with actual Gemini API call
        return np.random.rand(768).tolist()  # Replace with actual embedding dimensions

# Initialize Gemini embeddings
embeddings = GeminiEmbeddings()

# Load your knowledge base (e.g., a PDF file with answers)
def load_pdf(file_path):
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                documents.append(Document(text))
    return documents

class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}

documents = load_pdf('A3S.pdf')

# Split the documents into smaller chunks for better retrieval
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create embeddings for the documents using Gemini embeddings
embedded_docs = embeddings.embed_documents([doc.page_content for doc in docs])

# Convert embeddings to numpy array for FAISS
embedding_matrix = np.array(embedded_docs).astype(np.float32)

# Create a FAISS index
dimension = embedding_matrix.shape[1]  # The size of the embedding vector
index = faiss.IndexFlatL2(dimension)  # Use L2 distance (Euclidean distance)
index.add(embedding_matrix)

# Create a FAISS vector store
class FAISSWrapper:
    def __init__(self, index, docs):
        self.index = index
        self.docs = docs

    def as_retriever(self):
        return self

    def retrieve(self, query_embedding, k=5):
        D, I = self.index.search(np.array([query_embedding]).astype(np.float32), k)
        return [{'text': self.docs[i].page_content, 'score': float(D[0][j])} for j, i in enumerate(I[0])]

# Initialize FAISS vector store
vector_store = FAISSWrapper(index, docs)

def search_pdf(query):
    query_embedding = embeddings.generate_embedding(query)

    # Retrieve top results from FAISS index
    retriever_results = vector_store.retrieve(query_embedding, k=5)

    # Include chunks with scores
    chunks_with_scores = []
    for result in retriever_results:
        chunks_with_scores.append({
            'chunk': result['text'],
            'score': result['score']
        })
    
    return chunks_with_scores

def generate_response(query):
    # First, search within the PDF
    chunks_with_scores = search_pdf(query)

    if chunks_with_scores:
        responses = []
        for chunk in chunks_with_scores:
            text = chunk['chunk']
            score = chunk['score']
            response_obj = model.generate_content(text)
            response_text = response_obj.get("text", "") if isinstance(response_obj, dict) else str(response_obj)
            responses.append(f"Original Chunk: {text}\nResponse: {response_text}\nSimilarity Score: {score:.4f}\n{'-'*50}")
        return "\n".join(responses)
    else:
        # If no relevant chunks found, query Gemini directly
        response_obj = model.generate_content(query)
        response_text = response_obj.get("text", "") if isinstance(response_obj, dict) else str(response_obj)
        return f"Original Query: {query}\nResponse: {response_text}\nSimilarity Score: 0.0\n{'-'*50}"



@app.route('/')
def home():
    print("Home route accessed")
    return "Welcome to the Flask App!"

@app.route('/ask', methods=['POST'])
def ask():
    print("Ask route accessed")
    data = request.get_json()
    print(f"Received data: {data}")
    query = data.get('query')
    response_text = generate_response(query)
    print(f"Generated response: {response_text}")
    return Response(response_text, mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)



'''



@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query')
    response_text = generate_response(query)
    return Response(response_text, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
'''