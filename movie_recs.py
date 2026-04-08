import pymongo
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient 
from bson import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

client = MongoClient("mongodb://localhost:27017") 
db = client.project.movies 


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(text):
    return model.encode(text).tolist()

# count = 0
# for doc in db.find({"plot":{"$exists":True}}): 
#     emb = generate_embedding(doc['plot'])

#     db.update_one(
#         {'_id':ObjectId(f'{doc['_id']}')},
#         {"$set":{"embedding_hf":emb}}
#         )

#     count+=1
#     print(count,end=" ")


query = "space movie with alines" 
query_emb = generate_embedding(query) 

embeddings = [] 
ids = [] 

for doc in db.find({'plot':{'$exists':True}}): 
    embeddings.append(doc['embedding_hf']) 
    ids.append(doc['_id'])

X = np.array(embeddings) 
q = np.array(query_emb).reshape(1,-1) 

scores = cosine_similarity(q,X)[0]
top_k = scores.argsort()[-10:][::-1] 

print("Your request : ",query)

for i in top_k: 
    result = db.find_one({'_id':ObjectId(f'{ids[i]}')})
    print(result['title'])

