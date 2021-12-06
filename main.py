import pandas as pd
import numpy as np
import os
import re
import string
import random
import unicodedata
from string import punctuation
from string import digits
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

import spacy
from spacy.matcher import Matcher

from tools import print_issue, tfidf_embed, preprocess_manual, print_onissue_events, print_relatedissue_events, extract_entity_by_label,\
     extract_entity_by_index, spacy_tokenizer, print_issue, merge_sents

# 1. Read data
data_dir = 'data/'
filename_prefix = 'koreaherald_1517_'
df = []

for i in range(8):
    df.append(pd.read_json(os.path.join(data_dir, filename_prefix + str(i) + '.json')))
df = pd.concat(df)
df.reset_index(inplace=True)
df = df.rename(columns=dict(zip(df.columns,[df.columns[i].strip() for i in range(len(df.columns))])))
df.drop('index', inplace=True, axis=1)
df['agg_title_body'] = df['title'] + df['body']

# 2. Print Top 10 events in each year (2015 - 2017)
# Since the procedure is quasi-automatic.
print("\t2015: MERS, Birthrate decline, Seoul-Tokyo sex slavery issue, Vice President of Korean Air forcing cabin crew to disembark, Loan limit extension, Activist freed dogs, Yellow-dust, Audi Volkswagen, dead man body at Jeju island.")
print("\t2016: Pyeongyang broadcasting messages, Chickens death in Chungcheong, South Korea’s parliamentary election, Deoksu Palace restoration, contagious disease in pigs, National Research Center for Gifted and Talented Education,  peace treaty, Park Geun hye, citizen’s sugar consumption, North Korea missile test")
print("\t2017: Cold wave, dinosaur footprint, adhesive patch, lowest birthrate, the Stellar Daisy ship missing, bird flu outbreak, art connoisseurs and city officials gathered at Seoul City Hall,  Bang Tae-hyun taking a bribe,mosquito-borne virus, Pyeongyang broadcasting messages")
print()

# 3. Print On-Issue events
# Load large spacy model 
nlp = spacy.load('en_core_web_lg')

df_sum = pd.read_csv("data/data_sum.csv")
df_nk = pd.read_csv("data/df_nk_labeled")
top_nk = pd.read_csv("data/top_nk.csv")
df_politics = pd.read_csv("data/df_politics_labeled")
top_politics = pd.read_csv("data/top_politics.csv")

sents_nk = merge_sents(df_nk, 100).tolist()
sents_politics = merge_sents(df_politics, 1939).tolist()

sents_nk = list(map(preprocess_manual, sents_nk))
print_issue("North Korea Missile Test")
print_onissue_events(df_nk, label=100)
extract_entity_by_label(sents_nk, df_nk, 100)

sents_politics = list(map(preprocess_manual, sents_politics))
print_issue("President Park Geun-hye Middle-East tour")
print_onissue_events(df_politics, label=1939)
extract_entity_by_label(sents_politics, df_politics, 1939)

# 4. Print Related-Issue events
nk_related_issue_index = [22641, 18742, 9502, 19477, 13014]
sents_nk = [ ' '.join(spacy_tokenizer(s)) for s in df.iloc[nk_related_issue_index].sort_values(by=['time'])['body']]
print_issue("North Korea Missile Test")
print_relatedissue_events(df, nk_related_issue_index)
extract_entity_by_index(sents_nk, df, nk_related_issue_index)

p_related_issue_index = [8471,15914,17845,20506,22807]
sents_p = [ ' '.join(spacy_tokenizer(s)) for s in df.iloc[p_related_issue_index].sort_values(by=['time'])['body']]
print_issue("President Park Geun-hye Middle-East tour")
print_relatedissue_events(df, p_related_issue_index)
extract_entity_by_index(sents_nk, df, p_related_issue_index)