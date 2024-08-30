import pandas as pd
import re
import nltk
import numpy as np
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nltk.tokenize import word_tokenize
nltk.download('punkt')


train_csv_file = 'train.csv'
test_csv_file = 'test.csv'
train_data = pd.read_csv(train_csv_file)

dev_set_size = 7600
train_set = train_data.iloc[dev_set_size:]
dev_set = train_data.iloc[:dev_set_size]

train_sentences = train_set['Description'].tolist()
train_labels = train_set['Class Index'].tolist()

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    tokens = sentence.split()
    return ' '.join(tokens)

train_sentences_cleaned = [clean_sentence(sentence) for sentence in train_sentences]

train_sentences_tokenized = [word_tokenize(sentence) for sentence in train_sentences_cleaned]

print("Train Set Sentences (tokenized):")
print(train_sentences_tokenized[:2])
print("Train Set Labels:")
print(train_labels[:2])

print(f"train_dataset size: {len(train_sentences_cleaned)}")

all_words = [word for sentence in train_sentences_tokenized for word in sentence]
word_counts = Counter(all_words)

vocab = ['<UNK>', '<PAD>'] + [word for word, count in word_counts.items()]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

def sentence_to_indices(sentence, word2idx):
    return [word2idx.get(word, word2idx['<UNK>']) for word in sentence]

train_indices = [sentence_to_indices(sentence, word2idx) for sentence in train_sentences_tokenized]


print("Train Indices:")
print(train_indices[:2]) 
print("Train Set Labels:")
print(train_labels[:2])

train_all_sentences = [idx for sentence in train_indices for idx in sentence]

def create_ngrams(sequence, n):
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = sequence[i:i + n]
        ngrams.append(ngram)
    return torch.tensor(ngrams, dtype=torch.long)

ngram_size = 6
train_data = create_ngrams(train_all_sentences, ngram_size)

batch_size = 32
train_loader = DataLoader(TensorDataset(train_data[:100000]), batch_size=batch_size, shuffle=False)

print(f"Training Ngrams: {len(train_data)}")

print(train_data[:2])
print(train_data.dtype)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_glove_model(file_path):
    word_vectors = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            word_vectors[word] = vector

    return word_vectors

glove_file_path = 'glove.6B.100d.txt'
glove_dict = load_glove_model(glove_file_path)

def create_embedding_matrix(glove_dict):
    '''
    Creates a weight matrix of the words that are common in the GloVe vocab and
    the dataset's vocab. Initializes OOV words with a zero vector.
    '''
    weights_matrix = torch.randn((len(vocab), 100))
    words_found = 0
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = torch.tensor(glove_dict[word])
            words_found += 1
        except:
            pass
        
    return weights_matrix, words_found

embedding_matrix, words_found = create_embedding_matrix(glove_dict)

print("vocab size:", len(vocab))
print(f"words found in glove embedding: {words_found}")

class ELMo(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_matrix):
        super(ELMo, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding Layer, Default- freeze = True
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)

        # Forward Language Model
        self.lstm_forward1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_forward2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear_mode1 = nn.Linear(2*hidden_dim, vocab_size)

        # Backward Language Model
        self.lstm_backward1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_backward2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear_mode2 = nn.Linear(2*hidden_dim, vocab_size)

    def forward(self, input_data, mode):
        if mode == 1:
            forward_embed = self.embedding(input_data)
            forward_lstm1, _ = self.lstm_forward1(forward_embed) 
            forward_lstm2, _ = self.lstm_forward2(forward_lstm1)
            lstm_concat = torch.cat((forward_lstm1, forward_lstm2), dim=-1)
            output = self.linear_mode1(lstm_concat)
            return output
        
        elif mode == 2:
            backward_embed = self.embedding(input_data)
            backward_lstm1, _ = self.lstm_backward1(backward_embed) 
            backward_lstm2, _ = self.lstm_backward2(backward_lstm1)
            lstm_concat = torch.cat((backward_lstm1, backward_lstm2), dim=-1)
            output = self.linear_mode2(lstm_concat)
            return output

vocab_size =  len(vocab)
embedding_dim = 100
hidden_dim = 100

# Define the ELMo model
elmo = ELMo(vocab_size, embedding_dim, hidden_dim, embedding_matrix).to(device)

def train_mode1(model, mode, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for inputs in tqdm(train_loader):
        inputs = inputs[0]
        inputs = inputs.to(device)
        optimizer.zero_grad()

        if mode == 2:
            inputs = torch.flip(inputs, dims=[1])

        input_seq = inputs[:, :5]
        target_seq = inputs[:, 1:]
        outputs = model(input_seq, mode=mode)
        loss = criterion(outputs.permute(0, 2, 1), target_seq)  # Permute because outputs is (batch_size, seq_len, embed_dim) and target_Seq is (batch_size, seq_len).
        total_loss += loss.item()
        total_tokens += target_seq.numel()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / total_tokens

    return avg_loss

num_epochs = 4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(elmo.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train_loss_mode1 = train_mode1(elmo, 1, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} (Mode 1) - Train Loss: {train_loss_mode1:.4f}")

num_epochs = 4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(elmo.parameters(), lr=0.001)

for epoch in range(num_epochs):
    train_loss_mode1 = train_mode1(elmo, 2, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} (Mode 2) - Train Loss: {train_loss_mode1:.4f}")


PATH = 'pretrained_elmo'
torch.save(elmo.state_dict(), PATH)