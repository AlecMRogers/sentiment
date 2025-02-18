import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def run(data):
    # Load model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Convert text to embeddings
    label_embeddings = model.encode(["A negative review", "A positive review"])
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    # Find the best matching label for each document
    sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
    y_pred = np.argmax(sim_matrix, axis=1)

    return y_pred
