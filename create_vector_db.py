import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings


import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


with open("campus_info.txt", "r", encoding="utf-8") as file:
    data = file.read()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
texts = text_splitter.split_text(data)

embeddings = OpenAIEmbeddings()
text_vectors = embeddings.embed_documents(texts)

dimension = len(text_vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(text_vectors)


with open("faiss_index.pkl", "wb") as f:
    pickle.dump(index, f)

print("✅ Vector Database Created Successfully!")