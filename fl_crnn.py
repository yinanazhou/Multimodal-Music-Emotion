import json
import torch
from torchvision import transforms
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler, Dataset
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


# hyperparameters
BATCH_SIZE = 120  # feel free to change it
EPOCH = 500
TEST_SPLIT = 0.2
random_seed = 0
embedding_dim = 500
hidden_dim = 256

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
generator = torch.Generator().manual_seed(random_seed)


class MyDataset(Dataset):
    def __init__(self, audio_path, _lyrics, _labels, transform=None, idx=None):
        self.dataset = json.load(open(audio_path, 'r'))
        self.audio = np.array(self.dataset['melspectrogram'])
        self.lyrics = _lyrics
        self.labels = _labels
        if idx is not None:
            self.labels = self.labels[idx]
            self.audio = self.audio[idx]
            self.lyrics = self.lyrics[idx]
        self.transform = transform

    def __getitem__(self, index):
        _audio = self.audio[index].squeeze()
        _audio = Image.fromarray((_audio * 255).astype('uint8'), mode='L')
        # img = Image.fromarray(img.astype('uint8'), mode='L')
        if self.transform is not None:
            _audio = self.transform(_audio)
        return _audio, self.lyrics[index], self.labels[index]
        # return self.audio[index], self.lyrics[index], self.labels[index]

    def __len__(self):
        return self.audio.shape[0]


class Fusion(nn.Module):
    def __init__(self, input_size, _seq_len, _embedding_dim, num_layers, _hidden_dim, batch_size):
        super(Fusion, self).__init__()
        #  CNN
        self.conv0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 8), stride=(1, 1)),
            # nn.Tanh(),
            nn.Hardtanh(min_val=-0.5, max_val=0.5),
            nn.MaxPool2d(4, stride=4),
            nn.BatchNorm2d(32)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 8), stride=(1, 1)),
            # nn.Tanh(),
            nn.Hardtanh(min_val=-0.5, max_val=0.5),
            nn.MaxPool2d(4, stride=4),
            nn.BatchNorm2d(16),
        )
        # LSTM
        self.input_size = input_size
        self.embedding_dim = _embedding_dim
        self.num_layers = num_layers
        self.hidden_dim = _hidden_dim
        self._batch_size = batch_size

        self._embedding = nn.Embedding(input_size, _embedding_dim)
        self.lstm = nn.LSTM(16, _hidden_dim, num_layers, batch_first=True)    # , dropout=0.5
        self.hidden_state = self.init_hidden(batch_size)
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=_embedding_dim, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
                                   nn.ReLU(),
                                   nn.MaxPool2d((1, 2))
                                   )
        self._classifier0 = nn.Sequential(nn.Linear(in_features=16*8*78, out_features=100),
                                          # nn.Tanh(),
                                          nn.Hardtanh(min_val=-0.5, max_val=0.5),
                                          nn.Dropout(),
                                          nn.Linear(in_features=100, out_features=5))

        self._classifier1 = nn.Sequential(nn.Linear(in_features=_hidden_dim, out_features=100),
                                          # nn.Tanh(),
                                          nn.Hardtanh(min_val=-0.5, max_val=0.5),
                                          nn.Dropout(),
                                          nn.Linear(in_features=100, out_features=5))

    def forward(self, _audio, _lyric, hidden_state):
        # audio
        _audio = self.conv0(_audio)
        _audio = self.conv1(_audio)
        _audio_flat = _audio.view(-1, 16*8*78)
        _audio_out = F.softmax(self._classifier0(_audio_flat), dim=1)


        # lyric
        _lyric = _lyric.long()    # [batch_size, seq_len]
        _lyric = self._embedding(_lyric)  # [batch_size, seq_len, embedding_dim]
        _lyric = _lyric.permute(0, 2, 1).unsqueeze(2)
        _lyric = self.conv2(_lyric)
        _lyric = _lyric.squeeze(2).permute(0, 2, 1)
        _lyric = _lyric.contiguous().view(BATCH_SIZE, -1, 16)

        _lyric_lstm, hidden_state = self.lstm(_lyric, self.hidden_state)
        _lyric_flat = _lyric_lstm.contiguous().view(-1, self.hidden_dim)
        _lyric_out = F.softmax(self._classifier1(_lyric_flat), dim=1)  # [batch_size*seq_len, 1]
        _lyric_out = _lyric_out.view(BATCH_SIZE, -1, 5)
        _lyric_out = _lyric_out[:, -1]

        score = torch.cat((_audio_out, _lyric_out), dim=1)
        return score, hidden_state

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(DEVICE),
                  weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(DEVICE))
        return hidden


""" Load data"""
AUDIO_PATH = '../dataset/json/melspectrogram.json'
LYRIC_PATH = '../dataset/json/lyrics.json'

audio = json.load(open(AUDIO_PATH, 'r'))
lyrics = json.load(open(LYRIC_PATH, 'r'))

mel = audio["melspectrogram"]
corpus = lyrics["lyrics"]
labels = lyrics["labels"]

'''preprocess audio'''
transformer = transforms.Compose([transforms.ToTensor()])

'''preprocess lyrics'''
# tokenize & stop word removal & lemmatization
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

# get max lyric length
sentence_length = [len(sentence.split()) for sentence in corpus]
sentence_length_counts = Counter(sentence_length)   # lyric length counts
max_sen = max(sorted(sentence_length_counts.items()))
min_sen = min(sorted(sentence_length_counts.items()))

# build dict
words = [word.lower() for sentence in corpus for word in sentence.split(' ')]
various_words = list(set(words))    # different words
# various_words.remove('')            # remove null char
int_word = dict(enumerate(various_words, 1))    # word -> int
word_int = {w: int(i) for i, w in int_word.items()}  # int -> word

# map word to int
text_ints = []
for sentence in corpus:
    sample = list()
    for word in sentence.split():
        int_value = word_int[word]  # get the int corresponds to the word
        sample.append(int_value)
    text_ints.append(sample)

# pad all to same length
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

# convert data type
lyric_tensor = torch.from_numpy(lyric_ready)
labels = np.array(labels)
label_tensor = torch.from_numpy(labels)

'''load dataset'''
dataset = MyDataset(AUDIO_PATH, lyric_tensor, label_tensor, transform=transformer, idx=None)

'''split train&val, test'''
dataset_size = len(dataset)
test_length = int(dataset_size*TEST_SPLIT)
lengths = [dataset_size-test_length, test_length]
train_val_set, test_set = random_split(dataset, lengths, generator=generator)

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, generator=generator)

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

    train_set = MyDataset(AUDIO_PATH, lyric_tensor, label_tensor, transform=transformer, idx=train_ids)
    val_set = MyDataset(AUDIO_PATH, lyric_tensor, label_tensor, transform=transformer, idx=val_ids)

    trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=generator)
    valloader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, generator=generator)

    print('trainloader:', len(trainloader.dataset))
    print('valloader:', len(valloader.dataset))

    # Init the neural network
    network = Fusion(input_size=len(word_int) + 1, _seq_len=seq_len, _embedding_dim=embedding_dim,
                     _hidden_dim=hidden_dim, num_layers=1, batch_size=BATCH_SIZE).to(DEVICE)
    network.apply(reset_weights)
    hs = network.init_hidden(BATCH_SIZE)

    # Initialize optimizer
    optimizer = optim.Adam(network.parameters(), lr=1e-5)
    # optimizer = optim.SGD(network.parameters(), lr=lr)

    # Run the training loop for defined number of epochs
    for epoch in range(0, EPOCH):

        # Print epoch
        # print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        # current_loss = 0.0
        network.train()
        # Iterate over the DataLoader for training data
        for i, (audio, lyrics, targets) in enumerate(trainloader, 0):

            # Get inputs
            # inputs, targets = data
            audio = audio.to(DEVICE)
            lyrics = lyrics.to(DEVICE)
            targets = targets.to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs, hs = network(audio, lyrics, hs)

            # Compute loss
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
    save_path = f'model/model_fl_cr16-fold-{fold}.pth'
    save_opt = f'model/opt_fl_cr16_fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)
    torch.save(optimizer.state_dict(), save_opt)

    # Evaluationfor this fold
    correct, total = 0, 0
    network.eval()
    hs = network.init_hidden(BATCH_SIZE)
    with torch.no_grad():

        # Iterate over the val data and generate predictions
        for i, (audio, lyrics, targets) in enumerate(valloader, 0):
            # Get inputs
            # audio, lyrics, targets = data
            audio = audio.to(DEVICE)
            lyrics = lyrics.to(DEVICE)
            targets = targets.to(DEVICE)

            # Generate outputs
            outputs, hs = network(audio, lyrics, hs)

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

