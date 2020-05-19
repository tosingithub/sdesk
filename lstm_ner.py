# Tosin Adewumi
"""
Named Entity Recognition


"""

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from gensim.models import Word2Vec, KeyedVectors
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score


torch.manual_seed(1)
MAX_SEQ_LEN = 75
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
EPOCHS = 40
LR = 0.01
OUTPUT_FILE = "output_prints_lstm_1b_810.txt"
OUTPUT_FILE2 = "output_test_lstm_1b_810.txt"


class MakeSentence(object):
    """

    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w,p,t) for w, p, t in zip(s["Word"].values.tolist(),
                                                         s["POS"].values.tolist(),
                                                         s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

        def get_next(self):
            try:
                s = self.grouped["Sentence: {}".format(self.n_sent)]
                self.n_sent += 1
                return s
            except:
                return None


def prepare_sequence(seq, to_ix):
    """

    :param seq: sequence to encode
    :param to_ix: dictionary of enumerated vocab for encoding
    :return: return tensor of seq_to_ids
    """
    idxs = [to_ix[w] for w in seq]
    return idxs # torch.tensor(idxs, dtype=torch.long)


def pad_seq(sequence, tag2idx, seq_type='tok_ids', max_len=MAX_SEQ_LEN):
    padded_seq = []
    if seq_type == 'tok_ids':
        padded_seq.extend([0 for i in range(max_len)])              # initialize list with 0s to maximum seq length
    elif seq_type == 'tag_ids':
        padded_seq.extend([tag2idx['O'] for i in range(max_len)])   # initialize tag list with 'O's (NER- O: Other) to maximum seq length
    if len(sequence) > max_len:
        padded_seq[:] = sequence[:max_len]              # cut sequence longer than the maximum SEQ_LEN
    else:
        padded_seq[:len(sequence)] = sequence           # replace parts of default seq with values of original
    return padded_seq


class LSTMTagger(nn.Module):
    """

    """
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pretrained_model):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        # use pre-trained word vectors if chosen
        if pretrained_model == 0:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.weights = torch.FloatTensor(pretrained_model.wv.vectors)
            self.weights.requires_grad = False          # freeze weights - essential for optimal results
            self.word_embeddings = nn.Embedding.from_pretrained(self.weights)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence_tensor):
        embeds = self.word_embeddings(sentence_tensor)
        lstm_out, _ = self.lstm(embeds.view(len(sentence_tensor), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence_tensor), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores               # tensor with each row representing prediction of a token in a sequence


def list_into_chunks(list_name, n_chunks):
   for i in range(0, len(list_name), n_chunks):
       yield list_name[i:i + n_chunks]


def training_proc(sentences, labels, run_on_test=False, use_pretrained=False, do_graph=False):
    data_ids, tags_ids, tags_record, tags_record2, data_record = [], [], [], [], []
    # Shuffle & join everything first, so we don't have to do it later scattering order of masking
    train_data, t_data, train_tags, t_tags = train_test_split(sentences, labels, test_size=0.15, shuffle=True)
    train_data, val_data, train_tags, val_tags = train_test_split(train_data, train_tags, test_size=0.176, shuffle=True)
    sentences = train_data + val_data + t_data
    labels = train_tags + val_tags + t_tags

    for rg in range(len(sentences)):
        data_ids.append(prepare_sequence(sentences[rg].split(), word_to_ix))
        tags_ids.append(prepare_sequence(labels[rg], tag_to_ix))      # TODO: list comprehension
        data_record.append(sentences[rg].split())                     # keep original format for use later
        tags_record.append(labels[rg])                                # keep original format for metrics use

    # Padding before converting to tensors for batching
    data_inputs = [pad_seq(ii, tag_to_ix) for ii in data_ids]
    tags = [pad_seq(ti, tag_to_ix, seq_type='tag_ids') for ti in tags_ids]

    # masking to ignore padded items by giving floating value above 0.0 (1.0 in this case) to ids in input sequence
    attention_masks = [[float(i>0) for i in ii] for ii in data_inputs]

    # data set split (including masked data split if padding) | Padding (of both features & labels), masking & custom
    # loss are required if batching, which is faster but cumbersome. Otherwise, it's not compulsory but slower loading
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
    train_tags = torch.tensor(train_tags)
    val_inputs = torch.tensor(val_data)
    val_tags = torch.tensor(val_tags)
    test_inputs = torch.tensor(t_data)
    test_tags = torch.tensor(t_tags)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    test_masks = torch.tensor(test_masks)

    # pack inputs into tensordataset & dataloader
    train_tensor = TensorDataset(train_inputs, train_masks, train_tags)
    train_sampler = RandomSampler(train_tensor)
    train_dloader = DataLoader(train_tensor, sampler=train_sampler, batch_size=BATCH_SIZE)
    #print(next(iter(train_dloader)))
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
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), pretrained_model)
    loss_function = nn.CrossEntropyLoss()                           # custom loss needed if padding
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
        train_loss, train_steps = 0, 0                                  # to accumulate as we loop over data
        for batch in train_dloader:                                     # raw sentence/tags one by one instead of batch
            model.zero_grad()                                           # clear accumulated gradients
            batch_input_ids, batch_input_masks, batch_labels = batch
            batch_input_ids = batch_input_ids.to(device)
            batch_labels = batch_labels.to(device)
            # flatten the tensors to fit original code (Model accepts 1-D tensors)
            batch_input_ids = torch.flatten(batch_input_ids)
            batch_input_masks = torch.flatten(batch_input_masks)
            batch_labels = torch.flatten(batch_labels)
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

        # Validation
        model.eval()
        predictions = []
        counter = 0
        for batch in val_dloader:
            eval_loss, eval_acc, eval_steps = 0, 0, 0
            pred2 = []
            with torch.no_grad():                                       # no gradient compuation during validation
                vbatch_inputs, vbatch_masks, vbatch_tags = batch
                vbatch_inputs = vbatch_inputs.to(device)
                vbatch_tags = vbatch_tags.to(device)
                # flatten the tensors to fit original code (Model accepts 1-D tensors)
                vbatch_inputs = torch.flatten(vbatch_inputs)
                # vbatch_masks = torch.flatten(vbatch_masks)    # unused affects loss somewhat
                vbatch_tags = torch.flatten(vbatch_tags)
                tag_scores = model(vbatch_inputs)
                try:
                    loss = loss_function(tag_scores, vbatch_tags)
                except:
                    print("Invalid validation sample")
                    continue                                            # next iteration
                eval_loss += loss.item()
                eval_steps += 1
                # for computing our metrics the next few lines are necessary because we need to make sense
                # of the tag_scores (tensors) returned. In the tensor, the predicted tag is the maximum scoring tag,
                # hence we identify the corresponding index and map them to our tag/label dictionary.
                pred_values, indices = torch.max(tag_scores, dim=-1)    # flatten and return max values & their indices
                indices = indices.to("cpu")                             # copy to cpu before changing to numpy
                ind_num = indices.numpy()                               # change to numpy for easy manipulation
                pred1 = {k: v for k, v in tag_to_ix.items() if v in ind_num}  # dict comprehension: unique predictions
                for v in ind_num:                                       # loop over indices to return predicted keys
                    for k in pred1:
                        if pred1[k] == v:
                            pred2.append(k)     # TODO: list comprehension
                pred2 = list(list_into_chunks(pred2, MAX_SEQ_LEN))      # split flattened list into multiple lists of MAX_SEQ_LEN each
                # for accurate metrics, we need to remove padding
                for i in range(len(pred2)):
                    index_no = counter * BATCH_SIZE + i             # ensures loop through every value in list
                    max_len_original = len(true_labels[index_no])
                    pred2[i] = pred2[i][:max_len_original]          # remove padding from predicitions - to the length of original
                for i in range(len(pred2)):
                    predictions.append(pred2[i])
                counter += 1
        # Early stopping at the end of each epoch may be implemented here but left out for simplicity
        time_elapsed = time.time() - starttime                          # this is not necessary.
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
            s = f.write("Epoch: " + str(ep) + "\n" + "Predicted labels samples: " + str(predictions[:3]) + "\n" +
                        "Sentence samples: " + str(sentence_example[:3]) + "\n" +
                        "True labels samples: " + str(true_labels[:3]) + "\n" + "Time elapsed: " +
                        str(time_elapsed) + "\n" + "Validation loss: " + str(val_loss) + "\n" +
                        "Validation accuracy: " + str(eval_acc ) + "\n" + "F1: " + str(f1) + "\n" + "Precision: " +
                        str(precision) + "\n" + "Recall: " + str(recall) + "\n")
    if run_on_test:
        true_labels = rec_ttags  # to be used for metrics
        sentence_example = rec_tdata
        print("Test Set evaluation... ")
        model.eval()
        predictions = []
        counter = 0
        starttime = time.time()
        for batch in test_dloader:
            eval_loss, eval_acc, eval_steps = 0, 0, 0
            pred2 = []
            with torch.no_grad():                                       # no gradient compuation during evaluation
                tbatch_inputs, tbatch_masks, tbatch_tags = batch
                tbatch_inputs = tbatch_inputs.to(device)
                tbatch_tags = tbatch_tags.to(device)
                # flatten the tensors to fit original code (Model accepts 1-D tensors)
                tbatch_inputs = torch.flatten(tbatch_inputs)
                # tbatch_masks = torch.flatten(tbatch_masks)    # unused affects loss somewhat
                tbatch_tags = torch.flatten(tbatch_tags)
                tag_scores = model(tbatch_inputs)
                try:
                    loss = loss_function(tag_scores, tbatch_tags)
                except:
                    print("Invalid test sample")
                    continue                                            # next iteration
                eval_loss += loss.item()
                eval_steps += 1
                pred_values, indices = torch.max(tag_scores, dim=-1)    # flatten and return max values & their indices
                indices = indices.to("cpu")                             # copy to cpu before changing to numpy
                ind_num = indices.numpy()                               # change to numpy for easy manipulation
                pred1 = {k: v for k, v in tag_to_ix.items() if v in ind_num}  # dict comprehension: unique predictions
                for v in ind_num:                                       # loop over indices to return predicted keys
                    for k in pred1:
                        if pred1[k] == v:
                            pred2.append(k)     # TODO: list comprehension
                pred2 = list(list_into_chunks(pred2, MAX_SEQ_LEN))
                # for accurate metrics, we need to remove padding
                for i in range(len(pred2)):
                    index_no = counter * BATCH_SIZE + i             # ensures loop through every value in list
                    max_len_original = len(true_labels[index_no])
                    pred2[i] = pred2[i][:max_len_original]          # remove padding from predicitions - to the length of original
                for i in range(len(pred2)):
                    predictions.append(pred2[i])
                counter += 1
        time_elapsed = time.time() - starttime                          # this is not necessary.
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
            s = f.write("Test Set Evaluation \n" + "Predicted labels samples: " + str(predictions[:3]) + "\n" +
                        "Sentence samples: " + str(sentence_example[:3]) + "\n" +
                        "True labels samples: " + str(true_labels[:3]) + "\n" + "Time elapsed: " +
                        str(time_elapsed) + "\n" + "Test loss: " + str(test_loss) + "\n" +
                        "Test accuracy: " + str(eval_acc ) + "\n" + "F1: " + str(f1) + "\n" + "Precision: " +
                        str(precision) + "\n" + "Recall: " + str(recall) + "\n")


if __name__ == '__main__':
    data = pd.read_csv("ner_dataset.csv", encoding="latin1").fillna(method="ffill")
    get_sent = MakeSentence(data)           # instantiate sentence maker
    sentences = [" ".join(s[0] for s in sent) for sent in get_sent.sentences]   # concat data (originally in tokens) into sentences
    # sentences = sentences[:20]
    labels = [[s[2] for s in sent] for sent in get_sent.sentences]      # construct true labels for each sentence
    tags_vals = list(set(data["Tag"].values))                           # generate set of unique labels
    vocab = list(set(data["Word"].values))                              # generate vocab/unique data vales
    tag_to_ix = {t: i for i, t in enumerate(tags_vals)}                 # dictionary of labels/tags
    word_to_ix = {j: k for k, j in enumerate(vocab)}                    # dictionary of vocab/data
    # print(tag_to_ix)
    # tag_cnt = [t for t in data["Tag"] if t == "O"] for checking data distribution balance
    training_proc(sentences, labels, run_on_test=True, use_pretrained=True)
