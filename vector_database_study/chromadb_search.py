import chromadb

client = chromadb.PersistentClient(path="./my_vector_db")

collection = client.get_or_create_collection(name="my_documents", metadata={"hnsw:space": "cosine"})

get_result = collection.query(query_texts=["什么是深度学习？"])

print(get_result)
