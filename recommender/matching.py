import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load resources once ---
# Load model
with open("models/embedding_model_name.txt") as f:
    model_name = f.read().strip()
model = SentenceTransformer(model_name)

# Load project embeddings
project_embeddings = np.load("models/project_embeddings.npy")

# Load project IDs in order
with open("models/project_ids.pkl", "rb") as f:
    project_ids = pickle.load(f)

# --- Recommender function ---
def get_top_matches(proposal_text: str, top_n: int = 10) -> list[tuple[int, float]]:
    """
    Given a research proposal, return a list of (projectID, similarity score)
    for the top-N most similar Horizon projects.
    """
    input_vec = model.encode([proposal_text])
    sims = cosine_similarity(input_vec, project_embeddings)[0]

    top_indices = np.argsort(sims)[::-1][:top_n]
    top_project_ids = [(project_ids[i], float(sims[i])) for i in top_indices]

    return top_project_ids
