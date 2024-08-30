import pandas as pd
import re
import nltk
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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
test_data = pd.read_csv(test_csv_file)

dev_set_size = 7600
train_set = train_data.iloc[dev_set_size:]
dev_set = train_data.iloc[:dev_set_size] 

train_sentences = train_set['Description'].tolist()
train_labels = train_set['Class Index'].tolist()
dev_sentences = dev_set['Description'].tolist()
dev_labels = dev_set['Class Index'].tolist()
test_sentences = test_data['Description'].tolist()
test_labels = test_data['Class Index'].tolist()

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
    tokens = sentence.split()
    return ' '.join(tokens)

train_sentences_cleaned = [clean_sentence(sentence) for sentence in train_sentences]
dev_sentences_cleaned = [clean_sentence(sentence) for sentence in dev_sentences]
test_sentences_cleaned = [clean_sentence(sentence) for sentence in test_sentences]

train_sentences_tokenized = [word_tokenize(sentence) for sentence in train_sentences_cleaned]
dev_sentences_tokenized = [word_tokenize(sentence) for sentence in dev_sentences_cleaned]
test_sentences_tokenized = [word_tokenize(sentence) for sentence in test_sentences_cleaned]

train_data = pd.read_csv('train.csv')
length = []
for i in range(len(train_data['Description'])):
    sentence = re.findall(r"[\w']+|[.,!?;'-]", train_data['Description'][i])
    length.append(len(sentence))
length.sort()
max_seq_length = length[int(0.95*len(length))]+1


def pad_sequences(sentences, max_length):
    padded_sentences = []
    for sentence in sentences:
        if len(sentence) >= max_length:
            padded_sentences.append(sentence[:max_length])
        else:
            padded_sentences.append(sentence + ['<PAD>'] * (max_length - len(sentence)))
    return padded_sentences

train_sentences_padded = pad_sequences(train_sentences_tokenized, max_seq_length)
dev_sentences_padded = pad_sequences(dev_sentences_tokenized, max_seq_length)
test_sentences_padded = pad_sequences(test_sentences_tokenized, max_seq_length)

print("Train Set Sentences (Padded):")
print(train_sentences_padded[:2])
print("Train Set Labels:")
print(train_labels[:2])

all_words = [word for sentence in train_sentences_tokenized for word in sentence]
word_counts = Counter(all_words)

vocab = ['<UNK>', '<PAD>'] + [word for word, count in word_counts.items()]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}

def sentence_to_indices(sentence, word2idx):
    return [word2idx.get(word, word2idx['<UNK>']) for word in sentence]

train_indices_elmo = [sentence_to_indices(sentence, word2idx) for sentence in train_sentences_padded]
dev_indices_elmo = [sentence_to_indices(sentence, word2idx) for sentence in dev_sentences_padded]
test_indices_elmo = [sentence_to_indices(sentence, word2idx) for sentence in test_sentences_padded]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    def forward(self, input_data):
        forward_embed = self.embedding(input_data)
        forward_lstm1, _ = self.lstm_forward1(forward_embed)
        forward_lstm2, _ = self.lstm_forward2(forward_lstm1)

        input_data = torch.flip(input_data, dims=[1])
        backward_embed = self.embedding(input_data)
        backward_lstm1, _ = self.lstm_backward1(backward_embed)
        backward_lstm2, _ = self.lstm_backward2(backward_lstm1)
        
        # Flip bacward_lstm words, Concatenate forward1 and backward1 outputs
        backward_lstm1 = torch.flip(backward_lstm1, dims=[1])
        lstm_concat1 = torch.cat((forward_lstm1, backward_lstm1), dim=-1)
        
        # Concatenate forward2 and backward2 outputs
        backward_lstm2 = torch.flip(backward_lstm2, dims=[1])
        lstm_concat2 = torch.cat((forward_lstm2, backward_lstm2), dim=-1)

        # Concatenate forward and backward embeddings word by word
        embedding_concat = torch.cat((forward_embed, forward_embed), dim=-1)
        
        return embedding_concat, lstm_concat1, lstm_concat2
    

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 100

elmo = ELMo(vocab_size, embedding_dim, hidden_dim, torch.rand(vocab_size, embedding_dim)).to(device)

elmo.load_state_dict(torch.load('pretrained_elmo.zip', map_location=device))
elmo.to(device)

train_data_elmo = torch.tensor(train_indices_elmo, dtype=torch.long)
dev_data_elmo = torch.tensor(dev_indices_elmo, dtype=torch.long)
test_data_elmo = torch.tensor(test_indices_elmo, dtype=torch.long)

train_data_with_labels = TensorDataset(train_data_elmo[:100000], torch.tensor(train_labels[:100000], dtype=torch.long))
dev_data_with_labels = TensorDataset(dev_data_elmo[:5000], torch.tensor(dev_labels[:5000], dtype=torch.long))
test_data_with_labels = TensorDataset(test_data_elmo[:5000], torch.tensor(test_labels[:5000], dtype=torch.long))

batch_size = 32

train_loader = DataLoader(train_data_with_labels, batch_size=batch_size, shuffle=False)
dev_loader = DataLoader(dev_data_with_labels, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data_with_labels, batch_size=batch_size, shuffle=False)

print(train_data_with_labels[:1])

class LSTM(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, num_layers = 1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(1, dtype=torch.float32))


    def forward(self, embedding1, embedding2, embedding3):
        weights = nn.functional.softmax(self.weights, dim = 0)
        x = (embedding3*weights[0] + embedding1*weights[1] + embedding2*weights[2])*self.gamma
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        output, (hn, cn) = self.lstm(x, (h0, c0))                            # Initialize hidden state
        output_last = output[:, -1]                                          # Take the output from the last time step
        output_fc = self.fc(output_last)                                     # Pass it through the fully connected layer
        return output_fc

output_dim = train_data['Class Index'].unique().shape[0]
embedding_dim = 200
hidden_dim = 200

lstm = LSTM(output_dim, embedding_dim, hidden_dim).to(device)

def train_lstm(model,elmo_model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    elmo_model.eval()

    for input_seq, target_seq in tqdm(train_loader):
        input_seq = input_seq.to(device)
        target_seq = target_seq - 1
        target_seq = target_seq.to(device)
        optimizer.zero_grad()

        embedding1, embedding2, embedding3 = elmo_model(input_seq)

        outputs = model(embedding1, embedding2, embedding3)
        loss = criterion(outputs, target_seq) 
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        predicted = torch.argmax(outputs, dim = 1)
        correct += torch.sum(target_seq == predicted).item()
        total += target_seq.shape[0]
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss, correct / total


def evaluation_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)*100
    precision = precision_score(predicted_labels, true_labels, average='weighted')*100
    recall = recall_score(predicted_labels, true_labels, average='weighted')*100
    f1 = f1_score(predicted_labels, true_labels, average='weighted')*100
    confusion = confusion_matrix(predicted_labels, true_labels)
    print(f"Accuracy = {accuracy}")
    print(f"Recall = {recall}")
    print(f"F1-Score = {f1}")
    print(f"Precision = {precision}")
    print(f"Confusion Matrix: \n{confusion}\n")
    return


def evaluate_lstm(model, elmo_model, dev_loader):
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for input_seq, target_seq in tqdm(dev_loader):
            input_seq = input_seq.to(device)
            target_seq = target_seq - 1
            target_seq = target_seq.to(device)

            embedding1, embedding2, embedding3 = elmo_model(input_seq)

            outputs = model(embedding1, embedding2, embedding3)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(target_seq.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    return true_labels, predicted_labels


num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm.parameters(), lr = 1e-3)

for epoch in range(num_epochs):
    train_loss, accuracy = train_lstm(lstm, elmo, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} (Train) - Loss: {train_loss:.4f} Training Set Accuracy {accuracy*100}")

torch.save(lstm.state_dict(), 'classifier.pt')
true_labels, predicted_labels = evaluate_lstm(lstm, elmo, test_loader)
evaluation_metrics(true_labels=true_labels, predicted_labels=predicted_labels)



output_dim = train_data['Class Index'].unique().shape[0]
embedding_dim = 200
hidden_dim = 200
num_epochs = 10

lstm1 = LSTM(output_dim, embedding_dim, hidden_dim).to(device)
initial_weights = torch.randn(3, dtype = torch.float32, device = device)
initial_weights = nn.functional.softmax(initial_weights, dim = 0)
lstm1.weights = torch.nn.Parameter(initial_weights)
print(lstm1.weights, lstm1.gamma)

# Freeze the weights
lstm1.weights.requires_grad = False
lstm1.gamma.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm1.parameters(), lr = 1e-3)


print(f"For Initial Weights = {initial_weights}\n\n")

print("Training the Model\n")
for epoch in range(num_epochs):
    train_loss, accuracy = train_lstm(lstm1, elmo, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{num_epochs} (Train) - Loss: {train_loss:.4f} Training Set Accuracy {accuracy*100}")

print(lstm1.weights, lstm1.gamma)

print("Testing the Model on Test Set")
true_labels, predicted_labels = evaluate_lstm(lstm1, elmo, test_loader)
evaluation_metrics(true_labels=true_labels, predicted_labels=predicted_labels)