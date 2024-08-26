from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Set your API key for Gemini
os.environ['GOOGLE_API_KEY'] = "AIzaSyA3n-icZ6sukOHGIEiGeYkvV9wp8EFHr8o"

# Configure the API for Gemini
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')

# Define the GeminiEmbeddings class
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
        # Dummy embedding for demonstration
        return np.random.rand(768).tolist()  # Replace with actual embedding dimensions

# Initialize Gemini embeddings
embeddings = GeminiEmbeddings()

# Load your knowledge base (e.g., a text file with answers)
loader = TextLoader('abcdef.txt')
documents = loader.load()

# Split the documents into smaller chunks for better retrieval
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Create embeddings for the documents using Gemini embeddings
embedded_docs = embeddings.embed_documents(docs)

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

# Function to generate a response using LangChain and RAG
def generate_response(query):
    try:
        query_embedding = embeddings.generate_embedding(query)
        retriever_results = vector_store.retrieve(query_embedding)
        combined_text = " ".join([result['text'] for result in retriever_results])
        response = model.generate_content(combined_text)
        
        # Ensure the response is JSON serializable
        if hasattr(response, 'text'):  # Assuming response has a 'text' attribute
            return response.text
        else:
            return str(response)  # Fallback if 'text' attribute is not available

    except Exception as e:
        return str(e)  # Return error as string

# Initialize Flask app
app = Flask(__name__)

# Define a route for asking questions
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        if request.is_json:  # Handle API requests
            data = request.json
            query = data.get('query', '')
            if query:
                response_text = generate_response(query)
                return jsonify({'response': response_text})
            return jsonify({'error': 'No query provided'}), 400
        else:  # Handle form submissions
            query = request.form['query']
            response_text = generate_response(query)
            return render_template('index.html', query=query, response=response_text)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a simple home route for the frontend
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
