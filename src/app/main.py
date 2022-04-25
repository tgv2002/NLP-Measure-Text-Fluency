import pandas as pd
import re
import json
import joblib
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
from math import factorial, log
from itertools import combinations
from collections import Counter
import numpy as np
import gc
nltk.download('stopwords')
nltk.download('punkt')

from nltk.util import pad_sequence
from nltk.util import ngrams
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.translate.bleu_score import sentence_bleu

import torch
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Dropout, SimpleRNN, LSTM
from keras.callbacks import EarlyStopping
import tensorflow
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import streamlit as st

def is_valid_string(attribute_value):
    return not (attribute_value == None or pd.isnull(attribute_value) or \
                str(attribute_value) == "" or str(attribute_value) == "nan" or \
                len(attribute_value) == 0)

def initialize_models():
    with torch.no_grad():
        st.session_state.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        st.session_state.lm = BertForMaskedLM.from_pretrained('./models/BERT/')
        st.session_state.lm.eval()
    st.session_state.classifier = joblib.load('./models/rf_model.joblib')
    with open('./models/VOCAB.json', 'r') as f:
        st.session_state.VOCAB = json.load(f)
        st.session_state.TOTAL = sum(st.session_state.VOCAB.values())

def calculate_perplexity(_text):
    tokenize_inp = ["[CLS]"] + st.session_state.tokenizer.tokenize(_text.lower()) + ["[SEP]"]
    tensor_inp = torch.tensor([st.session_state.tokenizer.convert_tokens_to_ids(tokenize_inp)])
    with torch.no_grad():
        loss = st.session_state.lm(tensor_inp, labels=tensor_inp)[0]
    try:
        perplexity = np.exp(loss.detach().numpy())
        if float(perplexity) >= st.session_state.INF:
            perplexity = st.session_state.INF
        if float(perplexity) < st.session_state.EPSILON:
            perplexity = st.session_state.EPSILON
        denom = (perplexity ** (tensor_inp.shape[1] - 1))
        if denom >= st.session_state.INF:
            return st.session_state.INF, st.session_state.EPSILON
        if denom < st.session_state.EPSILON:
            return st.session_state.EPSILON, 0.9
        probability = 1 / (perplexity ** (tensor_inp.shape[1] - 1))
    except Exception as e:
        perplexity, probability = st.session_state.INF, st.session_state.EPSILON
    return perplexity, probability

def get_all_ngram_probabilities(model, given_text):
    text = given_text.lower()
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return [], st.session_state.INF
    all_ngram_probabilities, all_perplexities = [], []
    for sent in sentences:
        words = word_tokenize(sent)
        curr_ngrams = list(ngrams(words, n=3, pad_left=True, pad_right=True, left_pad_symbol="<s>", right_pad_symbol="</s>"))
        probabilities = [max(st.session_state.EPSILON, calculate_perplexity(' '.join(ngram))[1]) for ngram in curr_ngrams]
        all_ngram_probabilities.append(probabilities)
        try:
            perplexity = (1 / np.prod(np.asarray(probabilities))) ** (1 / len(probabilities))
        except:
            perplexity = st.session_state.INF
        all_perplexities.append(perplexity)
    return all_ngram_probabilities, sum(all_perplexities) / len(all_perplexities)

def get_prob_feature_vector(text, model):
    features = []
    all_ngram_probabilities, perplexity = get_all_ngram_probabilities(model, text) 
    all_ngram_probabilities = sorted(sum(all_ngram_probabilities, []))
    if len(all_ngram_probabilities) == 0:
        return [0 for _ in range(2*st.session_state.K)] + [st.session_state.INF]
    frequent_k, rarest_k = all_ngram_probabilities[-st.session_state.K:], all_ngram_probabilities[:st.session_state.K]
    if len(frequent_k) < st.session_state.K:
        median = frequent_k[len(frequent_k) // 2]
        for _ in range(st.session_state.K - len(frequent_k)):
            frequent_k.append(median)
        frequent_k = sorted(frequent_k)
    if len(rarest_k) < st.session_state.K:
        median = rarest_k[len(rarest_k) // 2]
        for _ in range(st.session_state.K - len(rarest_k)):
            rarest_k.append(median)
        rarest_k = sorted(rarest_k)
    features = features + frequent_k + rarest_k + [perplexity]
    return features
    
def get_SLOR(sentence, probability):
    if not is_valid_string(sentence):
        return 0
    sentence_tokens = word_tokenize(sentence.lower())[:40]
    if len(sentence_tokens) == 0:
        return 0
    term_1 = (1 / len(sentence_tokens)) * log(probability)
    unigram_prob = 1
    for word in sentence_tokens:
        if st.session_state.VOCAB.get(word) is not None:
            unigram_prob *= (st.session_state.VOCAB[word] / st.session_state.TOTAL)
        else:
            unigram_prob = 0
            break
    if unigram_prob < st.session_state.EPSILON:
        unigram_prob = st.session_state.EPSILON
    term_2 = log(unigram_prob)
    return term_1 - term_2

def get_text_label(_text):
    slor = get_SLOR(_text, calculate_perplexity(_text)[1])
    features = get_prob_feature_vector(_text, st.session_state.lm) + [slor]
    for j in range(len(features)):
        if str(features[j]) in ['inf', 'Infinity'] or features[j] >= st.session_state.INF:
            features[j] = st.session_state.INF
        if features[j] < st.session_state.EPSILON:
            features[j] = st.session_state.EPSILON
    return st.session_state.classifier.predict([features])[0]

PAGE_CONFIG = {'page_title':'NLP Project','layout':"wide"}

st.set_page_config(**PAGE_CONFIG)
 
if __name__ == '__main__':
    
    label_ids_to_names = {
        1: "Not Fluent", 2: "Neutral", 3: "Fluent"
    }
    
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = True
        st.session_state.EPSILON = 1e-20
        st.session_state.INF = 1e20
        st.session_state.K = 15
        initialize_models()
        
    st.title("Measuring Text Fluency")
    st.write('\n'*12)

    entered_text = st.text_area(label='Enter required text in the text area below, which would be used to label it with its most appropriate fluency label')

    col1, col2, col3 , col4, col5 = st.columns(5)
    with col1:
        pass
    with col2:
        pass
    with col4:
        pass
    with col5:
        pass
    with col3:
        submit = st.button('Submit')

    if submit:
        # Model is used for predicting label for the text here
        predicted_label = get_text_label(entered_text)
        st.markdown(f"### Text prediction: {label_ids_to_names[predicted_label]}")