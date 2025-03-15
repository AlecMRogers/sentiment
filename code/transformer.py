import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import classification_report


class Word2VecEmbedding:
    def __init__(self, sentences, embed_dim=100, model_path="word2vec.model"):
        """
        Initializes Word2Vec embedding model.
        """
        self.model_path = f"embeddings/{model_path}{embed_dim}"
        self.embed_dim  = embed_dim

        try:
            self.model = Word2Vec.load(self.model_path)
            print("Loaded pre-trained Word2Vec model.")
        except:
            print("Training new Word2Vec model...")
            if embed_dim == 100:
                # Step 1: Load Pre-trained Word2Vec Embeddings
                pretrained_vectors = KeyedVectors.load_word2vec_format('embeddings/enwiki_20180420_100d.txt',
                                                                       binary=False)
                # Step 2: Initialize a new Word2Vec model with the same settings
                self.model = Word2Vec(vector_size=pretrained_vectors.vector_size, window=5, min_count=1, workers=4)
                # Step 3: Build the Word2Vec vocabulary using the pre-trained model's vocab
                # self.model.build_vocab_from_freq( {word: 1 for word in pretrained_vectors.index_to_key} )  # Ensures it has all words
                # Step 4: Assign Pre-trained Vectors to Word2Vec Model
                self.model.wv = pretrained_vectors  # Replace the randomly initialized vectors
                # Step 5: Ensure Model is Ready for Use
                self.model.wv.fill_norms()  # Normalize if needed
                if (self.embed_dim != self.model.vector_size):
                    print("Error loading word2Vec model.")
            elif embed_dim == 300:
                pretrained_vectors = KeyedVectors.load_word2vec_format("embeddings/GoogleNews-vectors-negative300.bin",
                                                                       binary=True)
                self.model = Word2Vec(vector_size=pretrained_vectors.vector_size, window=5, min_count=1, workers=4)
                self.model.wv = pretrained_vectors
                self.model.wv.fill_norms()
                if (self.embed_dim != self.model.vector_size):
                    print("Error loading word2Vec model.")
            else:
                self.model = Word2Vec(sentences, vector_size=embed_dim, window=5, min_count=1, workers=4)
            self.model.save(self.model_path)
            print("Word2Vec model trained and saved.")



    def get_embedding(self, tokenized_sentences, max_len=None):
        """
        Converts tokenized sentences into Word2Vec embeddings.
        """
        if max_len is None:
            max_len = max(len(sentence) for sentence in tokenized_sentences)

        embeddings = []
        for sentence in tokenized_sentences:
            sentence_embeddings = []
            for token in sentence:
                if token in self.model.wv.key_to_index:
                    sentence_embeddings.append(self.model.wv[token])
                else:
                    sentence_embeddings.append(np.random.randn(self.embed_dim))

            while len(sentence_embeddings) < max_len:
                sentence_embeddings.append(np.zeros(self.embed_dim))

            embeddings.append(sentence_embeddings)

        return torch.from_numpy(np.array(embeddings, dtype=np.float32))
        #return torch.tensor(embeddings, dtype=torch.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        # normalize the positional encodings so that they do not overwhelm the data
        self.pe = self.pe / torch.norm(self.pe, dim=-1, keepdim=True)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))

        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        Q = self.W_Q(query)
        K = self.W_K(query)
        V = self.W_V(query)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        return attn_output, 0

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=6):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        Q = self.W_Q(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ff_dim=256):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=1)
        #self.attention = SingleHeadSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x,x,x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feedforward(x)
        x = self.norm2(x + ff_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, embed_dim=128, num_layers=1, ff_dim=256, num_classes=2):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(embed_dim, ff_dim) for _ in range(num_layers)]
        )

        self.classifier = nn.Linear(embed_dim, num_classes)  # Classification head

    def forward(self, x, mask=None):
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)

        x = x.mean(dim=0)  # Global average pooling over sequence
        x = self.classifier(x)  # Predict labels
        x = torch.sigmoid(x) # For BCE loss
        return torch.squeeze(x)

def train_model(model, train_embeddings, train_labels, num_epochs=5, lr=0.001, mask=None):
    """
    Trains the Transformer model using labeled training data.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss if
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_embeddings, mask)
        loss = loss_fn(output, train_labels)
        loss.backward()
        optimizer.step()
        print(f"Train Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    return output.detach().numpy()

def test_model(model, test_embeddings, test_labels):
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        output = model(test_embeddings, mask=None)
        loss = loss_fn(output, test_labels)
        print(f"Test Loss: {loss.item():.4f}")
        #test_probabilities = torch.softmax(test_logits, dim=1) # Convert logits to label predictions
        #output = torch.argmax(test_probabilities, dim=1)
    return output.numpy()

def create_mask(tokenized_sentences, max_len):
    mask = torch.zeros(len(tokenized_sentences), max_len, dtype=torch.bool)
    for i, sentence in enumerate(tokenized_sentences):
        mask[i, len(sentence):] = True
    return mask

def run(data, embed_dim = 100, num_epochs=100, lr=0.001):
    """
    Runs the Transformer model.
    """
    train_tokens = data["train"]["text"]
    train_labels = data["train"]["label"]
    train_labels = torch.tensor(train_labels, dtype=torch.float)  # Ensure labels are integral
    test_tokens  = data["test"]["text"]
    test_labels  = data["test"]["label"]
    test_labels  = torch.tensor(test_labels, dtype=torch.float)  # Ensure labels are integral

    rand_indx    = torch.randperm(len(train_labels))
    train_tokens = [train_tokens[i] for i in rand_indx]
    train_labels = train_labels[rand_indx]


    w2v = Word2VecEmbedding(sentences=train_tokens + test_tokens, embed_dim=embed_dim)

    max_len = max(max(len(s) for s in train_tokens), max(len(s) for s in test_tokens))
    train_embeddings = w2v.get_embedding(train_tokens, max_len=max_len)
    test_embeddings  = w2v.get_embedding(test_tokens, max_len=max_len)
    # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)

    train_embeddings = train_embeddings.permute(1, 0, 2) # Shape: (seq_length, batch_size, embedding_dim)
    test_embeddings  = test_embeddings.permute(1, 0, 2) # Shape: (seq_length, batch_size, embedding_dim)

    #train_padding_mask = create_mask(train_tokens, max_len)
    #test_padding_mask = create_mask(test_tokens, max_len)

    model = TransformerModel(embed_dim=embed_dim, num_classes=1) # for one-hot encoding, use len(set(train_labels.tolist()))

    # Train model on train_embeddings and train_labels
    train_model(model, train_embeddings, train_labels, num_epochs=num_epochs, lr=lr, mask=None)

    # Use test_embeddings for inference only
    output = test_model(model, test_embeddings, test_labels)
    output = np.round(output)
    return output


# Example standalone execution
if __name__ == "__main__":
    UseFullDataset =True

    if UseFullDataset:
        cache_file = "rottenTomatoes.data"

        # Load or cache the pre-trained Word2Vec model
        if os.path.exists(cache_file):
            print("Loading cached data...")
            data = torch.load(cache_file, weights_only=False)
        else:
            print("Downloading data...")
            data = load_dataset("rotten_tomatoes")
            torch.save(data, cache_file)

        # Have a look at the data
        print("\nInput: ", data["train"]["text"][0])
        print("\nOutput: ", data["train"]["label"][0])

        print("Running model...")
        y_actual = data["test"]["label"]
        y_pred = run(data)
        performance = classification_report(
            y_actual, y_pred,
            target_names=["Negative Review", "Positive Review"]
        )
        print(performance)
    else:
        sample_data = {
            "train": {
                "text": [["hello", "world"], ["transformers", "are", "powerful"]],
                "label": [0, 1]  # Corresponding labels
            },
            "test": {
                "text": [["deep", "learning"], ["attention", "mechanism"]]
            }
        }
        output = run(sample_data)
        print("Test Predictions:", output)

