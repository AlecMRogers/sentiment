import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=10000):
        print("Creating positional encoding ...")
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

class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SingleHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        # Scale dot product
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.embed_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.linear_q(query).view(batch_size, -1, 1, self.embed_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, 1, self.embed_dim).transpose(1, 2)
        value = value.view(batch_size, -1, 1, self.embed_dim).transpose(1, 2)

        attn_output, attn = self.scaled_dot_product_attention(query, key, value, mask)
        output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # Scale dot product
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        if key_padding_mask is not None:
            # Assuming key_padding_mask is a ByteTensor with shape [batch_size, seq_len] where padding elements are True
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size = query.size(0)

        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn = self.scaled_dot_product_attention(query, key, value, attn_mask, key_padding_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.final_linear(attn_output)

        return output, attn

class EmbeddingLayer(nn.Module):
    def __init__(self, embed_dim=100, sentences=None, model_path="word2vec.model"):
        super(EmbeddingLayer, self).__init__()
        """
        Initializes Word2Vec embedding model.
        """
        self.model_path          = f"embeddings/{model_path}{embed_dim}"
        self.embed_dim           = embed_dim
        self.norm                = nn.LayerNorm(embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.attention           = MultiHeadAttention(embed_dim, num_heads=4)
        self.embedding           = None
        self.mask                = None
        # Size of the embedding is Batch Size (2) X Sequence Length (3) X Embedding Dimension (100)
        self.max_len             = None #len(sentences)

        try:
            print("Loading pre-trained Word2Vec model...")
            self.model = Word2Vec.load(self.model_path)
        except:
            print("Training new Word2Vec model...")
            if embed_dim == 100:
                # Step 1: Load Pre-trained Word2Vec Embeddings
                pretrained_vectors = KeyedVectors.load_word2vec_format('embeddings/enwiki_20180420_100d.txt', binary=False)
                # Step 2: Initialize a new Word2Vec model with the same settings
                self.model = Word2Vec(vector_size=pretrained_vectors.vector_size, window=5, min_count=1, workers=4)
                # Step 3: Build the Word2Vec vocabulary using the pre-trained model's vocab
                #self.model.build_vocab_from_freq( {word: 1 for word in pretrained_vectors.index_to_key} )  # Ensures it has all words
                # Step 4: Assign Pre-trained Vectors to Word2Vec Model
                self.model.wv = pretrained_vectors  # Replace the randomly initialized vectors
                #self.model.wv.fill_norms()  # Normalize if needed
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
                self.model = Word2Vec(sentences=common_texts+sentences, vector_size=embed_dim, window=5, min_count=1, workers=4)
            self.model.save(self.model_path)
            print("Word2Vec model trained and saved.")
    def get_embedding(self, sentences):
        """
        Converts tokenized sentences into Word2Vec embeddings.
        """
        # tokenize the sentences, and compute the maximum number of tokens
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
        self.max_len = max(len(sentence) for sentence in tokenized_sentences)

        embeddings = []
        for sentence in tokenized_sentences:
            sentence_embeddings = []
            for token in sentence:
                if token in self.model.wv.key_to_index:
                    sentence_embeddings.append(self.model.wv[token])
                else:
                    sentence_embeddings.append(np.random.randn(self.embed_dim))

            while len(sentence_embeddings) < self.max_len:
                sentence_embeddings.append(np.zeros(self.embed_dim))

            embeddings.append(sentence_embeddings)

        e =  torch.from_numpy(np.array(embeddings, dtype=np.float32))
        return e.permute(1, 0, 2)  # Shape: (seq_length, batch_size, embedding_dim)
    def create_mask(self, tokenized_sentences):
        self.mask = torch.zeros(len(tokenized_sentences), self.max_len, dtype=torch.bool)
        for i, sentence in enumerate(tokenized_sentences):
            self.mask[i, len(sentence):] = True
    def forward(self, x, mask=None):
        # get embeddings from tokenized sentence for new x
        if x is not None:
            #print("Generating embedding ...")
            self.embedding = self.get_embedding(x)
            #print("Generating positional encoding ...")
            self.embedding = self.positional_encoding(self.embedding)
        x = self.embedding
        attn_output, _ = self.attention(x,x,x, mask)
        x = self.norm(x + attn_output)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, tf_dim=256):
        super(TransformerLayer, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, tf_dim),
            nn.ReLU(),
            nn.Linear(tf_dim, embed_dim)
        )

    def forward(self, x):
        tf_output = self.feedforward(x)
        x = self.norm(x + tf_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, embed_dim=128, sentences=None, tf_dim=256, num_classes=1):
        super(TransformerModel, self).__init__()
        self.embed_dim        = embed_dim
        self.embeddingLayer   = EmbeddingLayer(embed_dim, sentences=sentences)
        self.transformerLayer = TransformerLayer(embed_dim, tf_dim)
        self.classifier       = nn.Linear(embed_dim, num_classes)  # Classification head

    def forward(self, input):
        x = self.embeddingLayer(input)
        x = self.transformerLayer(x)
        x = x.mean(dim=0)  # Global average pooling over batch
        x = self.classifier(x)  # Predict labels
        x = F.sigmoid(x) # make sure output [0-1]
        return torch.squeeze(x)

    def runTrain(model, input, output, lr=0.001, batch_size=100):
        """
        Trains the Transformer model using labeled training data.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        model.train()
        allOutput = np.zeros_like(output)
        for i in range(0, len(input), batch_size):
            input_batch = input[i:i + batch_size]
            output_batch = output[i:i + batch_size]
            optimizer.zero_grad()
            eOutput = model(input_batch)
            loss = loss_fn(eOutput, output_batch)
            loss.backward()
            optimizer.step()
            allOutput[i:i + batch_size] = eOutput.detach().numpy()
        return list(np.round(allOutput).astype(int)), loss.item()

    def runTest(model, input, output, batch_size=100):
        loss_fn = nn.MSELoss()
        model.eval()
        allOutput = np.zeros_like(output)
        with torch.no_grad():
            for i in range(0, len(input), batch_size):
                input_batch = input[i:i + batch_size]
                output_batch = output[i:i + batch_size]
                eOutput = model(input_batch)
                loss = loss_fn(eOutput, output_batch)
                allOutput[i:i + batch_size] = eOutput.detach().numpy()
        return list(np.round(allOutput).astype(int)), loss.item()

    def plot_loss(self, train_losses, val_losses, test_losses):
        """
        Plots the training, validation, and test losses over time.
        """
        plt.figure(figsize=(10, 5))

        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='s')

        if test_losses:
            plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", marker='^')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over Time")
        plt.legend()
        plt.grid(True)

        # Save the plot as a high-resolution PNG (300 DPI)
        plt.savefig("transformer_loss.png", dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def run(data, embed_dim = 100, num_epochs=1, lr=0.0005, stopping_criterion=0.1):
        """
        Runs the Transformer model.
        """
        train_tokens = data["train"]["text"]
        train_labels = data["train"]["label"]
        train_labels = torch.tensor(train_labels, dtype=torch.float)

        validation_tokens  = data["validation"]["text"]
        validation_labels  = data["validation"]["label"]
        validation_labels  = torch.tensor(validation_labels, dtype=torch.float)

        test_tokens  = data["test"]["text"]
        test_labels  = data["test"]["label"]
        test_labels  = torch.tensor(test_labels, dtype=torch.float)

        rand_indx    = torch.randperm(len(train_labels))
        train_tokens = [train_tokens[i] for i in rand_indx]
        train_labels = train_labels[rand_indx]

        train_losses      = []
        validation_losses = []
        minValidationLoss = math.inf
        test_losses       = []

        # for one-hot encoding, use len(set(train_labels.tolist()))
        model = TransformerModel(embed_dim=embed_dim, sentences=train_tokens +validation_tokens+ test_tokens, tf_dim=256, num_classes=1)

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")

            output, train_loss = model.runTrain(train_tokens, train_labels, lr=lr)
            train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            output, validation_loss = model.runTest(validation_tokens, validation_labels)
            validation_losses.append(validation_loss)
            print(f"Validation Loss: {validation_loss:.4f}")

            output, test_loss = model.runTest(test_tokens, test_labels)
            test_losses.append(test_loss)
            print(f"Test Loss: {test_loss:.4f}")

            if validation_loss > minValidationLoss+stopping_criterion:
                print(f"Training stopped due to overfitting.")
                break
            if validation_loss < minValidationLoss:
                minValidationLoss = validation_loss

        # Plot the loss over time
        model.plot_loss(train_losses, validation_losses, test_losses)
        return output

# Standalone execution entry point
if __name__ == "__main__":
    UseFullDataset = True

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
        y_pred = TransformerModel.run(data, num_epochs=100)
        performance = classification_report(
            y_actual, y_pred,
            target_names=["Negative Review", "Positive Review"]
        )
        print(performance)
    else:
        data = {
            "train": {
                "text": ["hello world", "transformers are powerful"], # nPercepts = 3
                "label": [0, 1]
            },
            "validation": {
                "text": ["foo bar", "hello world"],            # nPercepts = 2
                "label": [1, 0]
            },
            "test": {
                "text": ["deep learning", "attention mechanism"], # nPercepts = 2
                "label": [1, 1]
            }
        }
        output = TransformerModel.run(data, num_epochs=100)
        print("Test Predictions:", output)

