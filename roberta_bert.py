# Tosin Adewumi

"""


"""

import torch
from scipy.special import softmax
import logging
from pytorch_transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM,\
    RobertaForSequenceClassification
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM,\
    BertForNextSentencePrediction, BertForQuestionAnswering, BertForMultipleChoice,\
    BertForSequenceClassification, BertForPreTraining, BertForTokenClassification


def roberta_prelude(model_type, model_download, string1, string2="", output_para2=[]):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param output_para2
    :return:
    """
    robertatok = RobertaTokenizer.from_pretrained(model_download)
    robertamod = model_type.from_pretrained(model_download)
    robertamod.eval()  # disable dropout for evaluation - only forward pass
    # Encode a pair of sentences for contradiction: 0
    token_ids_sent1 = robertatok.encode(string1)
    if string2 is not "":
        print("RoBERTa prelude tokenizing 2 sentences...")
        token_ids_sent2 = robertatok.encode(string2)
        token_ids = robertatok.add_special_tokens_sentences_pair(token_ids_sent1, token_ids_sent2) # try encode add special token
    else:
        print("RoBERTa prelude tokenizing 1 sentence...")
        token_ids = robertatok.add_special_tokens_single_sentence(token_ids_sent1)
    input_ids = torch.tensor(token_ids).unsqueeze(0)  # Batch size 1
    if len(output_para2) > 0:
        print("RoBERTa prelude running model with 2nd parameter labels...")
        labels = torch.tensor(output_para2).unsqueeze(0)
        output = robertamod(input_ids, labels=labels)
    else:
        if model_type is RobertaForMaskedLM:
            print("RoBERTa prelude running with masked lm labels...")
            output = robertamod(input_ids, masked_lm_labels=input_ids)
        else:
            print("RoBERTa prelude running model without labels...")
            output = robertamod(input_ids)

    return output, robertatok


def roberta_model(model_type, model_download, string1, string2):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param output_para2:
    :return:
    """
    # Basic RoBERTa program to tokenize 2 inputs, use base RoBERTa model and output tensor
    output, robertatok = roberta_prelude(model_type, model_download, string1, string2)
    #print("Last hidden state (output)", output)
    print("Length of output:", len(output))
    print("Len of Output [0] - The last hidden state = the 1st element of the output tuple (batch_size):", len(output[0]))
    print("Len of Output [0][0] - sequence length (tokens) of last hidden state:", len(output[0][0]))
    print("Len of Output [0][0][0] - hidden size (units) of the last hidden state:", len(output[0][0][0]))
    print("Len of Output [1]:", len(output[1]))


def roberta_seq_classification(model_type, model_download, string1, string2, output_para2):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param output_para2
    :return:
    """
    # For Sequence Classification (Regression) Case
    output, robertatok = roberta_prelude(model_type, model_download, string1, string2, output_para2)
    loss, classification_scores = output[:2]
    print("Loss ", loss)
    print("Classification scores ", classification_scores)


def roberta_seq_classification2(model_type, model_download, string1, string2, output_para2):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param output_para2:
    :return:
    """
    output, robertatok = roberta_prelude(model_type, model_download, string1, string2, output_para2)
    loss, classification_scores = output[:2]
    print("Loss ", loss)
    print("Classification scores ", classification_scores)

    #   For 2 sentences Sequence Classification Case
    # labels: 2 - for classification loss - Cross Entropy, 1 for regression - MeanSquare
    #labels = torch.tensor([2]).unsqueeze(0)
    #output = model(input_ids, labels=labels)


def roberta_masked(model_type, model_download, string1, string2=""):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param output_para2:
    :return:
    """
    output, robertatok = roberta_prelude(model_type, model_download, string1, string2)
    loss, prediction_scores = output[:2]
    print("Loss ", loss)
    print("Prediction scores ", prediction_scores)

    #   For Masked LM Case


def roberta_mask(model_type, model_download, string1, string2="", output_para2=[], topk=20):
    """
    # Prediction of a masked token in RoBERTa
    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param output_para2:
    :param topk:
    :return:
    """
    output, robertatok = roberta_prelude(model_type, model_download, string1, string2, output_para2)
    loss, prediction_scores = output[:2]
    tokenizer = RobertaTokenizer.from_pretrained(model_download)
    tokenized_text = tokenizer.tokenize(string1)
    masked_index = tokenized_text.index('<mask>')+1
    # labels: 2 for classification loss - Cross Entropy, 1 for regression - MeanSquare
    print("Cross Entropy Loss ", loss.item())
    predicted_index = torch.argmax(prediction_scores[0, masked_index]).item()
    print("Most likely Index", predicted_index)
    predicted_token = tokenizer.decode(predicted_index)
    print("Most likely Token", predicted_token)

    predicted_k_indexes = torch.topk(prediction_scores[0, masked_index], topk)
    predicted_logits_list = predicted_k_indexes[0]
    predicted_indexes_list = predicted_k_indexes[1]
    # Employ post processing to remove unnecessary tokens from the result list.
    for i, item in enumerate(predicted_indexes_list):
        the_index = predicted_indexes_list[i].item()
        print("Words and logits", tokenizer.decode(the_index), predicted_logits_list[i].item())


def bert_prelude(model_type, model_download, string1="", string2="", masked_index=8, multilist=[]):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param masked_index:
    :param multilist:
    :return:
    """
    if model_download is "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(model_download, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_download)
    tokenized_text = tokenizer.tokenize(string1)
    if model_type is BertForMaskedLM:
        tokenized_text[masked_index] = '[MASK]'
    # Define sentence A (0) and B(1) indices associated to 1st and 2nd sentences
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1] #replace with [id for id in range(len(tokenized_text))]
    if model_type is BertForMultipleChoice:
        tokens_tensor = torch.tensor([tokenizer.encode(s) for s in multilist]).unsqueeze(0) # Batch size 1, 2 choices
        segments_tensors = torch.tensor([segments_ids])
    else:
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
    # load pretrained weights & set evaluation mode to deactivate dropout  - important for reproducible results
    model = model_type.from_pretrained(model_download)
    model.eval()
    # if cuda available then put inputs on cuda
    return tokens_tensor, segments_tensors, tokenized_text, model


def bert_masked(model_type, model_download, string1, string2, masked_index):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param masked_index:
    :return:
    """
    tokens_tensor, segments_tensors, tokenized_text, model = bert_prelude(model_type, model_download, string1, string2, masked_index)
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    # confirm prediction
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = BertTokenizer.from_pretrained(model_download).convert_ids_to_tokens([predicted_index])[0]
    print("Masked Token:",predicted_token)
    #assert predicted_token == "henson"


def bert_mask(model_type, model_download, string1, string2, masked_index):
    """

    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :param masked_index:
    :return:
    """
    # Predicting Masked Token in BERT
    tokens_tensor, segments_tensors, tokenized_text, model = bert_prelude(model_type, model_download, string1, string2, masked_index)
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    # loss, prediction_scores = predictions[:2] # confirm
    # print("Cross Entropy Loss ", loss.item())
    predicted_index = torch.argmax(predictions[0][0, masked_index]).item()
    predicted_token = BertTokenizer.from_pretrained(model_download).convert_ids_to_tokens([predicted_index])[0]
    print("Prediceted Index", predicted_index)
    print("Prediceted Token", predicted_token)

    predicted_k_indexes = torch.topk(predictions[0][0, masked_index], k=20)
    predicted_logits_list = predicted_k_indexes[0]
    predicted_indexes_list = predicted_k_indexes[1]
    # Employ post processing to remove unnecessary tokens from the result list.
    for i, item in enumerate(predicted_indexes_list):
        the_index = predicted_indexes_list[i].item()
        print("Words and logits", BertTokenizer.from_pretrained(model_download).convert_ids_to_tokens(the_index),
              predicted_logits_list[i].item())


def bert_next_sentence(model_type, model_download, string1, string2):
    """
    # Predicting Next Sentence in BERT
    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :return:
    """
    tokens_tensor, segments_tensor, tokenized_text, model = bert_prelude(model_type, model_download, string1, string2)
    with torch.no_grad():
        next_sent_class_logits = model(tokens_tensor, segments_tensor)
    print(next_sent_class_logits)
    #max(logits) (NB: 0 is True but 1 is False)  # Some take the 1st element but Thomas Wolf says max
    #res = softmax(next_sent_class_logits.squeeze().tolist())


def bert_qa(model_type, model_download, string1, string2):
    """
    # Question Answering in BERT
    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :return:
    """
    tokens_tensor, segments_tensor, tokenized_text, model = bert_prelude(model_type, model_download, string1, string2)
    with torch.no_grad():
        start_logits, end_logits = model(tokens_tensor, segments_tensor)
    print("Start logits", start_logits)
    print("End logits", end_logits)


def bert_seq_classification(model_type, model_download, string1, string2):
    """
    # BERT for Sequence Classification
    :param model_type:
    :param model_download:
    :param string1:
    :param string2:
    :return:
    """
    tokens_tensor, segments_tensor, tokenized_text, model = bert_prelude(model_type, model_download, string1, string2)
    labels = torch.tensor([0]).unsqueeze(0)     # for computing classification(>1)/regression(1) loss
    with torch.no_grad():
        seq_class_logits = model(tokens_tensor, segments_tensor, labels=labels)
    loss, logits = seq_class_logits[:2]
    print("Model Loss", loss)
    print("Sequence Logits", logits)


def bert_multi_choice(model_type, model_download, multilist):
    """
    # BERT for Multiple Choice
    :param model_type:
    :param model_download:
    :param multilist:
    :return:
    """
    tokens_tensor, segments_tensor, tokenized_text, model = bert_prelude(model_type, model_download, multilist=multilist)
    labels = torch.tensor(0).unsqueeze(0)     # for computing classification(>1)/regression(1) loss
    with torch.no_grad():
        multichoice_logits = model(tokens_tensor, labels=labels)
    loss, logits = multichoice_logits[:2]
    print("Model Loss", loss)
    print("MultiChoice Classification Logits", logits)


if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)
    #bert_multi_choice(BertForMultipleChoice, "bert-base-uncased", ["Hello, my dog is cute","Hello, my cat is amazing"])

    #bert_seq_classification(BertForSequenceClassification,"bert-base-uncased","[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]","")

    #bert_qa(BertForQuestionAnswering,"bert-base-uncased","[CLS] Where is the bank ? [SEP] The bank is in paris . [SEP]","")
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    # text = " Who was Jim Henson ? [SEP] The bank is in paris . [SEP]"
    # text = "[CLS] Where is the bank ? [SEP] The bank is in paris . [SEP]"

    #bert_next_sentence(BertForNextSentencePrediction,"bert-base-uncased","[CLS] Where is the bank ? [SEP] The bank is in paris . [SEP]","")
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    # text = " Who was Jim Henson ? [SEP] The bank is in paris . [SEP]"
    # text = "[CLS] Where is the bank ? [SEP] The bank is in paris . [SEP]"

    bert_mask(BertForMaskedLM,"bert-base-uncased","[CLS] I know my aunt is healthy [SEP] since she eats a lot of vegetables [SEP]","",masked_index=4)

    #bert_masked(BertForMaskedLM,"bert-base-uncased","[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]","",masked_index=8)

    #roberta_mask(RobertaForMaskedLM,"roberta-base","I know my <mask> is healthy since she eats a lot of vegetables.","",topk=20)
    # text = 'I know my aunt is <mask> since she eats a lot of vegetables.'  # Great example / Adjectives
    # text = 'I know my <mask> is healthy since she eats a lot of vegetables.'  # Biased to women / Nouns

    #roberta_masked(RobertaForMaskedLM,"roberta-base","Roberta is a heavily optimized version of BERT.","Roberta is not very optimized.")
    #roberta_seq_classification(RobertaForSequenceClassification,"roberta-base","Hello, my dog is cute","",[1])

    #roberta_seq_classification2(RobertaForSequenceClassification,"roberta-base","Roberta is a heavily optimized version of BERT.","Roberta is not very optimized.",[2])
    # token_ids_sent1 = tokenizer.encode('')
    # token_ids_sent2 = tokenizer.encode('')
    # robertamod.predict('mnli', tokens).argmax()  # 0: contradiction

    #roberta_model(RobertaModel,"roberta-base","Roberta is a heavily optimized version of BERT.","Roberta is not very optimized.")