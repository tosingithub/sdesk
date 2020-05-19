# Tosin Adewumi
"""
Sentiment Analysis


"""
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from gensim.models import Word2Vec, KeyedVectors
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from collections import Counter


DIR_SA = "data_sentiments"
MAX_SEQ_LEN = 250
torch.manual_seed(1)
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
EPOCHS = 20
LR = 0.0001
OUTPUT_FILE = "output_sa_lstm_bw_810.txt"
OUTPUT_FILE2 = "output_satest_lstm_bw_810.txt"


def preprocess_pandas(data, columns):
    # word_tokens = []
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
    return data


def seq_to_ix(seq, to_ix):
    """

    :param seq: sequence to encode
    :param to_ix: dictionary of enumerated vocab for encoding
    :return: return tensor of seq_to_ids
    """
    idxs = [to_ix[w] for w in seq]
    return idxs # torch.tensor(idxs, dtype=torch.long)


def encode_sents(all_sents):
    encoded_sents = list()
    for sent in all_sents:
        encoded_sent = list()
        for word in sent.split():
            if word not in words_to_ix.keys():
                encoded_sent.append(0)                  # put 0 for out of vocab words
            else:
                encoded_sent.append(words_to_ix[word])
        encoded_sents.append(encoded_sent)
    return encoded_sents


def pad_seq(sequence, tag2idx, seq_type='tok_ids', max_len=MAX_SEQ_LEN):
    padded_seq = []
    if seq_type == 'tok_ids':
        padded_seq.extend([0 for i in range(max_len)])              # initialize list with 0s to maximum seq length
    elif seq_type == 'tag_ids':
        padded_seq.extend([tag2idx[0] for i in range(max_len)])
    if len(sequence) > max_len:
        padded_seq[:] = sequence[:max_len]              # cut sequence longer than the maximum SEQ_LEN
    else:
        padded_seq[:len(sequence)] = sequence           # replace parts of default seq with values of original
    return padded_seq


class SABiLSTM(nn.Module):
    """
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, pretrained_model=0, output_size=1, n_layers=1): # drop_prob=0.5
        super(SABiLSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        if pretrained_model == 0:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.weights = torch.FloatTensor(pretrained_model.wv.vectors)
            self.weights.requires_grad = False          # freeze weights - essential for optimal results
            self.embedding = nn.Embedding.from_pretrained(self.weights)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True) # dropout=drop_prob,
        # self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size()
        embedd = self.embedding(x)
        lstm_out, hidden = self.lstm(embedd)

        # stack up the lstm output
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)

        # dropout and fully connected layers
        #out = self.dropout(lstm_out)
        out = self.fc1(lstm_out)
        #out = self.dropout(out)
        out = self.fc2(out)
        sig_out = self.sigmoid(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out

    # def init_hidden(self, batch_size):
    #     """Initialize Hidden STATE"""
    #     # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
    #     # initialized to zero, for hidden state and cell state of LSTM
    #     weight = next(self.parameters()).data
    #
    #     # if (train_on_gpu):
    #     #     hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
    #     #               weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
    #     # else:
    #     hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
    #
    #     return hidden


def training_proc(sentences, labels, run_on_test=False, use_pretrained=False, do_graph=False):
    data_ids, tags_ids, tags_record, tags_record2, data_record = [], [], [], [], []
    train_data, t_data, train_tags, t_tags = train_test_split(sentences, labels, test_size=0.15, shuffle=True)
    train_data, val_data, train_tags, val_tags = train_test_split(train_data, train_tags, test_size=0.176, shuffle=True)

    # Convert pandas Series to lists in this case
    train_data = train_data.tolist()
    val_data = val_data.tolist()
    t_data = t_data.tolist()
    train_tags = train_tags.tolist()
    val_tags = val_tags.tolist()
    t_tags = t_tags.tolist()
    sentences = train_data + val_data + t_data
    labels = train_tags + val_tags + t_tags

    # transform sequence items to ids
    # data_ids = seq_to_ix(sentences, sents_to_ix)
    data_ids = encode_sents(sentences)
    tags_ids = seq_to_ix(labels, tag_to_ix)
    data_record = sentences                     # keep original format for use later
    tags_record = labels                        # keep original format for metrics use

    # Padding before converting to tensors for batching
    data_inputs = [pad_seq(ii, tag_to_ix) for ii in data_ids]
    # tags = [pad_seq(ti, tag_to_ix, seq_type='tag_ids') for ti in tags_ids]    # not needed in SA
    tags = tags_ids

    # masking to ignore padded items by giving floating value above 0.0 (1.0 in this case) to ids in input sequence
    attention_masks = [[float(i>0) for i in ii] for ii in data_inputs]

    train_data, t_data, train_tags, t_tags = train_test_split(data_inputs, tags, test_size=0.15, shuffle=False)       # for 70:15:15
    temp_tr = train_data              # needed in the mask section
    train_data, val_data, train_tags, val_tags = train_test_split(train_data, train_tags, test_size=0.176, shuffle=False)  # for 70:15:15
    train_masks, test_masks, i1, i2 = train_test_split(attention_masks, data_inputs, test_size=0.15, shuffle=False)
    train_masks, val_masks, i3, i4 = train_test_split(train_masks, temp_tr, test_size=0.176, shuffle=False)

    # Now split the record sets according to the above also
    rec_traindata, rec_tdata, rec_traintags, rec_ttags = train_test_split(data_record, tags_record, test_size=0.15, shuffle=False)
    rec_traindata, rec_valdata, rec_traintags, rec_valtags = train_test_split(rec_traindata, rec_traintags, test_size=0.176, shuffle=False)

    # convert data to tensors
    train_inputs = torch.tensor(train_data)
    train_tags = torch.FloatTensor(train_tags)          # using FloatTensor cos BCE loss requires it
    val_inputs = torch.tensor(val_data)
    val_tags = torch.FloatTensor(val_tags)
    test_inputs = torch.tensor(t_data)
    test_tags = torch.FloatTensor(t_tags)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    test_masks = torch.tensor(test_masks)

    # pack inputs into tensordataset & dataloader
    train_tensor = TensorDataset(train_inputs, train_masks, train_tags)
    train_sampler = RandomSampler(train_tensor)
    train_dloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=BATCH_SIZE)
    # print(next(iter(train_dloader)))
    val_tensor = TensorDataset(val_inputs, val_masks, val_tags)
    val_sampler = SequentialSampler(val_tensor)
    val_dloader = DataLoader(val_tensor, sampler=val_sampler, batch_size=BATCH_SIZE)
    test_tensor = TensorDataset(test_inputs, test_masks, test_tags)
    test_sampler = SequentialSampler(test_tensor)
    test_dloader = DataLoader(test_tensor, sampler=test_sampler, batch_size=BATCH_SIZE)

    pretrained_model = 0        # Initiliaze to 0 for status check when called in model
    if use_pretrained:
        print("Using Pretrained vectors...")
        # pretrained_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        pretrained_model = KeyedVectors.load("../env_wv/word2vec_m5_s300_w8_s1_h0_n5_i10")

    model = SABiLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(words_to_ix)+1, pretrained_model) #, len(tag_to_ix), pretrained_model)
    loss_function = nn.BCELoss()                            # BCELoss (if sigmoid applied to output) or BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Check if GPU available, move model & data to GPU if so
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda")
        model.to(device)

    # training & validation
    true_labels = rec_valtags              # to be used for metrics
    sentence_example = rec_valdata
    train_loss_list, val_loss_list = [], []        # for graphs
    val_acc_list, test_acc_list = [], []
    starttime = time.time()                                             # not necessary
    for ep in trange(EPOCHS, desc="Epoch"):
        model.train()                                                   # necessary to begin training
        #h = model.init_hidden(BATCH_SIZE)
        train_loss, train_steps = 0, 0                                  # to accumulate as we loop over data
        for batch in train_dloader:
            model.zero_grad()                                           # clear accumulated gradients
            batch_input_ids, batch_input_masks, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)
            tag_scores = model(batch_input_ids)
            try:
                loss = loss_function(tag_scores, batch_labels)
            except:
                print("Invalid training sample")
                continue                                                # next iteration
            loss.backward()                                             # back-propagate
            optimizer.step()                                            # update gradients
            train_loss += loss.item()
            train_steps += 1
        print("Train loss: {}".format(train_loss / train_steps))        # mean loss over iteration per epoch
        #
        # Validation
        model.eval()
        predictions = []
        num_correct, counter, acc_add = 0, 0, 0
        for batch in val_dloader:
            eval_loss, eval_acc, eval_steps = 0, 0, 0
            with torch.no_grad():                                       # no gradient computation during validation
                vbatch_inputs, vbatch_masks, vbatch_tags = batch
                vbatch_inputs = vbatch_inputs.to(device)
                vbatch_tags = vbatch_tags.to(device)
                tag_scores = model(vbatch_inputs)
                try:
                    loss = loss_function(tag_scores, vbatch_tags)
                except:
                    print("Invalid validation sample")
                    continue                                            # next iteration
                eval_loss += loss.item()
                eval_steps += 1
                pred = torch.round(tag_scores.squeeze())  # rounds or floors predicted output probabilities to 1 or 0
                # correct_tensor = pred.eq(vbatch_tags.float().view_as(pred))   # comparison returns True or False for each
                # correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
                pred = pred.to("cpu")                                   # copy to cpu before numpy manipulation
                pred = pred.numpy()
                predictions.extend(pred)
        time_elapsed = time.time() - starttime                          # this is not necessary.
        predictions = torch.FloatTensor(predictions)                    # back to tensor for metrics cos of type of true tags
        val_loss = eval_loss / eval_steps
        eval_acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        print("Validation loss: {}".format(val_loss))
        print("Validation Accuracy: {}".format(eval_acc))
        print("F1 score: {}".format(f1))
        print("Precision score: {}".format(precision))
        print("Recall score: {}".format(recall))
        train_loss_list.append(train_loss/train_steps)                  # for graphs
        val_loss_list.append(val_loss)
        val_acc_list.append(eval_acc)
        with open(OUTPUT_FILE, "a+") as f:                                  # not needed: store outputs in a file
            s = f.write("Epoch: " + str(ep) + "\n" + "Predicted labels samples: " + str(predictions[:10]) + "\n" +
                        "Sentence samples: " + str(sentence_example[:10]) + "\n" +
                        "True labels samples: " + str(true_labels[:10]) + "\n" + "Time elapsed: " +
                        str(time_elapsed) + "\n" + "Validation loss: " + str(val_loss) + "\n" +
                        "Validation accuracy: " + str(eval_acc ) + "\n" + "F1: " + str(f1) + "\n" + "Precision: " +
                        str(precision) + "\n" + "Recall: " + str(recall) + "\n")
    # graphs can be plotted here
    # test set evaluation if selected
    if run_on_test:
        true_labels = rec_ttags  # to be used for metrics
        sentence_example = rec_tdata
        print("Test Set evaluation... ")
        model.eval()
        predictions = []
        starttime = time.time()
        for batch in test_dloader:
            eval_loss, eval_acc, eval_steps = 0, 0, 0
            with torch.no_grad():                                       # no gradient compuation during evaluation
                tbatch_inputs, test_masks, tbatch_tags = batch
                tbatch_inputs = tbatch_inputs.to(device)
                tbatch_tags = tbatch_tags.to(device)
                tag_scores = model(tbatch_inputs)
                try:
                    loss = loss_function(tag_scores, tbatch_tags)
                except:
                    print("Invalid test sample")
                    continue                                            # next iteration
                eval_loss += loss.item()
                eval_steps += 1
                pred = torch.round(tag_scores.squeeze())  # rounds or floors predicted output probabilities to 1 or 0
                # correct_tensor = pred.eq(vbatch_tags.float().view_as(pred))   # comparison returns True or False for each
                # correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
                pred = pred.to("cpu")                                   # copy to cpu before numpy manipulation
                pred = pred.numpy()
                predictions.extend(pred)
        time_elapsed = time.time() - starttime                          # this is not necessary.
        predictions = torch.FloatTensor(predictions)                    # back to tensor for metrics cos of type of true tags
        test_loss = eval_loss / eval_steps
        eval_acc = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        print("Test loss: {}".format(test_loss))
        print("Test Accuracy: {}".format(eval_acc))
        print("F1 score: {}".format(f1))
        print("Precision score: {}".format(precision))
        print("Recall score: {}".format(recall))
        with open(OUTPUT_FILE2, "a+") as f:                                  # not needed: store outputs in a file
            s = f.write("Test Set Evaluation \n" + "Predicted labels samples: " + str(predictions[:10]) + "\n" +
                        "Sentence samples: " + str(sentence_example[:10]) + "\n" +
                        "True labels samples: " + str(true_labels[:10]) + "\n" + "Time elapsed: " +
                        str(time_elapsed) + "\n" + "Test loss: " + str(test_loss) + "\n" +
                        "Test accuracy: " + str(eval_acc ) + "\n" + "F1: " + str(f1) + "\n" + "Precision: " +
                        str(precision) + "\n" + "Recall: " + str(recall) + "\n")


if __name__ == "__main__":
    # get data, pre-process and split
    data = pd.read_csv(DIR_SA+"/imdbmoviedata.tsv", delimiter='\t', header=None)
    # data = data[:200]
    data.columns = ['Id', 'Class', 'Sentence']
    data['index'] = data.index                                          # add new column index
    columns = ['index', 'Id', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)                               # pre-process
    tags_vals = list(set(data["Class"].values))                           # generate set of unique labels
    unique_sents = list(set(data["Sentence"].values))                     # generate unique data/sentence values
    tag_to_ix = {t: i for i, t in enumerate(tags_vals)}                   # dictionary of labels/tags
    sents_to_ix = {j: k for k, j in enumerate(unique_sents)}
    all_sents = " ".join(data["Sentence"].values)
    all_tokens = all_sents.split()
    token_count = Counter(all_tokens)                                     # dict of unique words & their total count
    total_no_words = len(token_count)
    sorted_words = token_count.most_common(total_no_words)
    words_to_ix = {s: t+1 for t, (s, u) in enumerate(sorted_words)}       # (s,u) to match s (key) to t (value)
    training_proc(data['Sentence'], data['Class'], run_on_test=True, use_pretrained=True)
