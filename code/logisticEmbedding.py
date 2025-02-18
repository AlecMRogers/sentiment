from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def run(data):
    # Load model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Convert text to embeddings
    train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
    test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

    # Train a Logistic Regression on our train embeddings
    clf = LogisticRegression(random_state=42)
    clf.fit(train_embeddings, data["train"]["label"])

    # Test
    y_pred = clf.predict(test_embeddings)

    return y_pred