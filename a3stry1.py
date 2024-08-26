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

# Define a class to handle Gemini embeddings
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
    print(f"Loaded {len(documents)} pages from the PDF.")
    return documents

class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = {}

documents = load_pdf('A3S.pdf')

# Split the documents into smaller chunks for better retrieval
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"Generated {len(docs)} chunks from the PDF.")

# Create embeddings for the documents using Gemini embeddings
embedded_docs = embeddings.embed_documents([doc.page_content for doc in docs])
print(f"Generated embeddings for {len(embedded_docs)} chunks.")
print(f"Sample embedding: {embedded_docs[0]}")

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
        results = [{'text': self.docs[i].page_content, 'score': float(D[0][j])} for j, i in enumerate(I[0])]
        
        # Sort results by score ascending (lower distance means higher similarity)
        sorted_results = sorted(results, key=lambda x: x['score'])
        
        # Apply a threshold for similarity score (L2 distance in this case)
        threshold = 200  # Adjust this threshold as needed
        filtered_results = [res for res in sorted_results if res['score'] < threshold]
        
        return filtered_results

# Initialize FAISS vector store
vector_store = FAISSWrapper(index, docs)

def search_pdf(query):
    query_embedding = embeddings.generate_embedding(query)
    retriever_results = vector_store.retrieve(query_embedding, k=5)

    print(f"Retrieved {len(retriever_results)} results from FAISS.")
    for result in retriever_results:
        print(f"Chunk: {result['text'][:100]}... Score: {result['score']}")
    
    return retriever_results

def generate_response(query):
    chunks_with_scores = search_pdf(query)
    responses = []

    if chunks_with_scores:
        for chunk in chunks_with_scores:
            text = chunk['text']
            score = chunk['score']
            response_obj = model.generate_content(text)
            response_text = response_obj.get("text", "") if isinstance(response_obj, dict) else str(response_obj)
            responses.append(f"Original Chunk: {text}\nResponse: {response_text}\nSimilarity Score: {score:.4f}\n{'-'*50}")
    else:
        response_obj = model.generate_content(query)
        response_text = response_obj.get("text", "") if isinstance(response_obj, dict) else str(response_obj)
        responses.append(f"Original Query: {query}\nResponse: {response_text}\nSimilarity Score: 0.0\n{'-'*50}")

    return "\n".join(responses)

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
