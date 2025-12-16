from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import os

mtcnn = MTCNN(image_size=160, margin=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def embed(filePath):
    img = Image.open(filePath).convert("RGB")  
    face = mtcnn(img)
    if face is None:
        raise ValueError(f"❌ No face detected in {filePath}!")
    embedding = model(face.unsqueeze(0).to(device))
    return embedding.detach().cpu().numpy().flatten().tolist()

# -----------------------------
# ChromaDB setup (cosine similarity)
# -----------------------------
client = chromadb.PersistentClient(path="./face_db")
collection = client.get_or_create_collection(
    name="students",
    metadata={"hnsw:space": "cosine"}   # important fix ✅
)

# -----------------------------
# Store embedding with name
# -----------------------------
def store_embedding(filePath, name):
    emb = embed(filePath)
    collection.add(
        ids=[name],            # use name as unique ID
        embeddings=[emb],      # vector
        metadatas=[{"name": name}]
    )
    print(f"✅ Stored embedding for {name}")


def remove_student(name):
    """Remove a student's embeddings by their name (ID)."""
    try:
        collection.delete(ids=[name])
        print(f"🗑️ Removed embeddings for {name}")
    except Exception as e:
        print(f"⚠️ Error removing {name}: {e}")


def search_embedding(filePath, top_k=3):
    """Return top_k matches sorted by cosine similarity."""
    query_emb = np.array(embed(filePath)).reshape(1, -1)

    # Get embeddings + metadata from Chroma
    results = collection.get(include=["embeddings", "metadatas"])

    names = [m["name"] for m in results["metadatas"]]
    db_embs = np.array(results["embeddings"])

    # Compute cosine similarity manually ✅
    sims = cosine_similarity(query_emb, db_embs)[0]

    # Sort by similarity high → low
    matches = sorted(zip(names, sims), key=lambda x: x[1], reverse=True)

    return matches[:top_k]



if __name__ == "__main__":
    folder_path = "bande"

    # Loop through all files in bande folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jpg"):  # Sirf jpg images lo
            name = os.path.splitext(file_name)[0]  # File ka naam -> person ka naam
            img_path = os.path.join(folder_path, file_name)

            # Store embedding
            store_embedding(img_path, name.capitalize())