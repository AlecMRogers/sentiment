import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def run(data):
    # Load model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Convert text to embeddings
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    # Average the embeddings of all documents in each target label
    df = pd.DataFrame(np.hstack([train_embeddings, np.array(data["train"]["label"]).reshape(-1, 1)]))
    averaged_target_embeddings = df.groupby(768).mean().values

    # Find the best matching embeddings between evaluation documents and target embeddings
    sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)

    return y_pred
