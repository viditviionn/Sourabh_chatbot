from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import faiss
import pdfplumber

app = Flask(__name__)

# Set your API key for Gemini
os.environ['GOOGLE_API_KEY'] = "AIzaSyA3n-icZ6sukOHGIEiGeYkvV9wp8EFHr8o"

# Configure the API for Gemini
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-pro')

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
        return np.random.rand(768).tolist()

embeddings = GeminiEmbeddings()

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

# Update the path to your PDF file
documents = load_pdf('A3S.pdf')

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

embedded_docs = embeddings.embed_documents(docs)

embedding_matrix = np.array(embedded_docs).astype(np.float32)

dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

class FAISSWrapper:
    def __init__(self, index, docs):
        self.index = index
        self.docs = docs

    def as_retriever(self):
        return self

    def retrieve(self, query_embedding, k=5):
        D, I = self.index.search(np.array([query_embedding]).astype(np.float32), k)
        return [{'text': self.docs[i].page_content, 'score': float(D[0][j])} for j, i in enumerate(I[0])]

vector_store = FAISSWrapper(index, docs)

def generate_response(query):
    print(" I am in post method")
    query_embedding = embeddings.generate_embedding(query)

    retriever_results = vector_store.retrieve(query_embedding, k=5)

    filtered_results = [result for result in retriever_results if result['score'] < 0.9]
    print(filtered_results)
    responses = []

    if filtered_results:
        print("Filtered results found.")
        i=0
        for result in filtered_results:
            text = result['text']
            score = result['score']
            print(f"Result {i}:")
            
            i=i+1
            print(text)

            response_obj = model.generate_content(text)
            response_text = response_obj.get("text", "") if isinstance(response_obj, dict) else str(response_obj)
            
            responses.append({'response': response_text, 'text': text, 'score': score})
    else:
        print("No Filtered Result Found")
        response_obj = model.generate_content(query)
        response_text = response_obj.get("text", "") if isinstance(response_obj, dict) else str(response_obj)
        
        responses = [{'response': response_text, 'text': 'No relevant document text', 'score': 0.0}]
    
    return responses




@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/ask', methods=['POST'])
def ask():
    print("i am in ask post method")
    data = request.get_json()
    query = data.get('query')
    responses = generate_response(query)
    return responses
    return jsonify(responses)
    

if __name__ == '__main__':
    app.run(debug=True)
