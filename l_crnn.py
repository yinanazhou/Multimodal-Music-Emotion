import json
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler, Dataset
from sklearn.model_selection import KFold
from collections import Counter
import nltk
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

print('hello, world')
TEST_SPLIT = 0.2
random_seed = 0
BATCH_SIZE = 120  # feel free to change it
EPOCH = 500
embedding_dim = 300
hidden_dim = 128

class LoadData:
    def __init__(self, _lyrics, _labels, idx=None):
        self.lyrics = _lyrics
        self.labels = _labels
        if idx is not None:
            self.lyrics = self.lyrics[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.lyrics[index], self.labels[index]


class LSTM(nn.Module):
    def __init__(self, input_size, _embedding_dim, num_layers, _hidden_dim, batch_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.embedding_dim = _embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = _hidden_dim
        self._batch_size = batch_size

        self._embedding = nn.Embedding(input_size, _embedding_dim)
        self._conv0 = nn.Sequential(nn.Conv1d(in_channels=_embedding_dim, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
                                    nn.ReLU(),
                                    nn.MaxPool2d((1, 2))
                                    )

        # self.lstm = nn.LSTM(_embedding_dim, _hidden_dim, num_layers, batch_first=True, dropout=0.5)    # , dropout=0.5
        self._lstm = nn.LSTM(16, _hidden_dim, batch_first=True)
        self.hidden_state = self.init_hidden(batch_size)

        self._classifier = nn.Sequential(nn.Linear(in_features=_hidden_dim, out_features=64),
                                         nn.Tanh(),
                                         #  nn.Hardtanh(min_val=-0.5, max_val=0.5),
                                         nn.Dropout(0.5),
                                         nn.Linear(in_features=64, out_features=5))

    def forward(self, x, hidden_state):
        x = x.long()    # [batch_size, seq_len]
        x = self._embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1).unsqueeze(2)   # [batch_size, embedding_dim, 1, seq_len]

        x = self._conv0(x)
        x = x.squeeze(2)
        # print(x.shape)
        x = x.permute(0, 2, 1)
        x = x.view(BATCH_SIZE, -1, 16)

        lstm_out, hidden_state = self._lstm(x, self.hidden_state)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = F.softmax(self._classifier(lstm_out), dim=1)  # [batch_size*seq_len, 1]
        out = out.view(BATCH_SIZE, -1, 5)
        out = out[:, -1]
        return out, hidden_state

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(DEVICE),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(DEVICE))
        return hidden


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

'''load data'''
lyrics_path = '../dataset/json/lyrics.json'
lyric_dataset = json.load(open(lyrics_path, 'r'))

corpus = lyric_dataset["lyrics"]
labels = lyric_dataset["labels"]

'''tokenize & stop word removal & lemmatization'''
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
WORD_COUNT = []
def preprocess(lyric):
    # tokenize
    word_list = nltk.word_tokenize(lyric)
    # stop word removal
    word_list = [word for word in word_list if not word in STOPWORDS]
    # length of lyrics
    word_count = len(word_list)
    # lemmatization
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
    return lemmatized_output, word_count

# preprocess corpus
for i in range(len(corpus)):
    corpus[i], count = preprocess(corpus[i])
    WORD_COUNT.append(count)

'''get max lyric length'''
sentence_length = [len(sentence.split()) for sentence in corpus]
sentence_length_counts = Counter(sentence_length)   # lyric length counts
max_sen = max(sorted(sentence_length_counts.items()))
min_sen = min(sorted(sentence_length_counts.items()))

'''build dict'''
words = [word.lower() for sentence in corpus for word in sentence.split(' ')]
various_words = list(set(words))    # different words
# various_words.remove('')            # remove null char
int_word = dict(enumerate(various_words, 1))    # word -> int
word_int = {w: int(i) for i, w in int_word.items()}  # int -> word

'''map word to int'''
text_ints = []
for sentence in corpus:
    sample = list()
    for word in sentence.split():
        int_value = word_int[word]  # get the int corresponds to the word
        sample.append(int_value)
    text_ints.append(sample)

'''pad all to same length'''
def reset_text(text, seq_len):
    _dataset = np.zeros((len(text), seq_len))
    for index, sentence in enumerate(text):
        if len(sentence) < seq_len:
            _dataset[index, :len(sentence)] = sentence   # pad
        else:
            _dataset[index, :] = sentence[:seq_len]  # cut
    return _dataset

seq_len = int((max_sen[0]-min_sen[0])/2)
lyric_ready = reset_text(text_ints, seq_len=seq_len)

'''convert data type'''
lyric_tensor = torch.from_numpy(lyric_ready)
labels = np.array(labels)
label_tensor = torch.from_numpy(labels)

'''split train&val, test'''
# dataset = TensorDataset(lyric_tensor, label_tensor)
dataset = LoadData(lyric_tensor, label_tensor, idx=None)
dataset_size = len(dataset)
test_length = int(dataset_size*TEST_SPLIT)
lengths = [dataset_size-test_length, test_length]

train_val_set, test_set = random_split(dataset, lengths, generator=torch.Generator().manual_seed(random_seed))

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


"""Model"""
def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


k_folds = 5
results = {}
# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_ids, val_ids) in enumerate(kfold.split(train_val_set)):
    print('--------------------------------')
    print(f'FOLD {fold}')

    train_set = LoadData(lyric_tensor, label_tensor, idx=train_ids)
    val_set = LoadData(lyric_tensor, label_tensor, idx=val_ids)

    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print('trainloader:', len(trainloader.dataset))
    print('valloader:', len(valloader.dataset))

    # Init the neural network
    network = LSTM(input_size=len(word_int)+1, _embedding_dim=embedding_dim, _hidden_dim=hidden_dim, num_layers=1, batch_size=BATCH_SIZE).to(DEVICE)
    network.apply(reset_weights)
    hs = network.init_hidden(BATCH_SIZE)

    # Initialize optimizer
    # optimizer = optim.Adam(network.parameters(), lr=1e-5)
    optimizer = optim.SGD(network.parameters(), lr=1e-5)

    # Run the training loop for defined number of epochs
    for epoch in range(0, EPOCH):

        # Print epoch
        # print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        # current_loss = 0.0
        network.train()
        # Iterate over the DataLoader for training data
        for i, (inputs, targets) in enumerate(trainloader, 0):

            # Get inputs
            # inputs, targets = data
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            # print(targets)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs, hs = network(inputs, hs)

            # Compute loss
            # loss = F.cross_entropy(outputs, targets, reduction='sum')
            loss = F.cross_entropy(outputs, targets.long(), reduction='sum')
            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            # current_loss += loss.item()
            # if i % 200 == 0:
            #     print('Loss after mini-batch %5d: %.3f' %
            #           (i + 1, current_loss / 500))
            #     current_loss = 0.0

    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting validating')

    # Saving the model
    save_path = f'model/model_l_cr01-fold-{fold}.pth'
    save_opt = f'model/opt_l_cr01-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)
    torch.save(optimizer.state_dict(), save_opt)

    # Evaluationfor this fold
    correct, total = 0, 0
    network.eval()
    hs = network.init_hidden(BATCH_SIZE)
    print('valloader before with torch.no_grad():', len(valloader.dataset))
    with torch.no_grad():
        print('in with torch.no_grad():')
        # Iterate over the test data and generate predictions
        for i, (inputs, targets) in enumerate(valloader, 0):
            print('in valloader loop')
            print('valloader in for loop:', len(valloader.dataset))
            # Get inputs
            # inputs, targets = data
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            # Generate outputs
            outputs, hs = network(inputs, hs)

            # Set total and correct
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
_sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    _sum += value
print(f'Average: {_sum / len(results.items())} %')