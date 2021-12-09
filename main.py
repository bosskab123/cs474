import pandas as pd
import os
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tools import print_issue, tfidf_embed, preprocess_manual, print_onissue_events, print_relatedissue_events, extract_entity_by_label,\
     extract_entity_by_index, spacy_tokenizer, print_issue, merge_sents, related_issue_event, most_related_docs

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

# Load large spacy model 
nlp = spacy.load('en_core_web_lg')

# Get from running summary.ipynb and change name
df_sum = pd.read_csv("data/data_sum.csv")

# Get from running task2_plot_eps_nk.ipynb
df_nk = pd.read_csv("data/df_nk_labeled")
top_nk = pd.read_csv("data/top_nk.csv")
df_politics = pd.read_csv("data/df_politics_labeled")
top_politics = pd.read_csv("data/top_politics.csv")

# Concatenate north korea sentence in the same label group id (label 100)
sents_nk = merge_sents(df_nk, 100).tolist()
# Concatenate politics sentence in the same label group id (label 1939)
sents_politics = merge_sents(df_politics, 1939).tolist()

# 3.1 Print On-Issue events : North Korea Missile Test
sents_nk = list(map(preprocess_manual, sents_nk))
print_issue("North Korea Missile Test")
print_onissue_events(df_nk, label=100)
extract_entity_by_label(sents_nk, df_nk, 100)

# 3.2 Print On-Issue events : President Park Geun-hye Middle-East tour
sents_politics = list(map(preprocess_manual, sents_politics))
print_issue("President Park Geun-hye Middle-East tour")
print_onissue_events(df_politics, label=1939)
extract_entity_by_label(sents_politics, df_politics, 1939)

# 4.1 Print Related-Issue events : North Korea Missile Test
nk_vectorizer = CountVectorizer(tokenizer=spacy_tokenizer)
nk_data_vectorized = nk_vectorizer.fit_transform(df_nk['agg_title_body'])
nk_lda_components = 50
nk_lda = LatentDirichletAllocation(n_components=nk_lda_components)
nk_lda.fit(nk_data_vectorized)
nk_doc_topic_dist = pd.DataFrame(nk_lda.transform(nk_data_vectorized))

nk_related_issue_index = related_issue_event(5, ['missile'], df_nk, nk_doc_topic_dist, nk_lda, nk_vectorizer, n_top_words=10)
sents_nk = [ ' '.join(spacy_tokenizer(s)) for s in df.iloc[nk_related_issue_index].sort_values(by=['time'])['body']]
print_issue("North Korea Missile Test")
print_relatedissue_events(df, nk_related_issue_index)
extract_entity_by_index(sents_nk, df, nk_related_issue_index)

# 4.2 Print Related-Issue events : President Park Geun-hye Middle-East tour
p_vectorizer = CountVectorizer(tokenizer=spacy_tokenizer)
p_data_vectorized = p_vectorizer.fit_transform(df_politics['agg_title_body'])
p_lda_components = 50
p_lda = LatentDirichletAllocation(n_components=p_lda_components)
p_lda.fit(p_data_vectorized)
p_doc_topic_dist = pd.DataFrame(p_lda.transform(p_data_vectorized))

p_related_issue_index = related_issue_event(5, ['middle','east'], df_politics, p_doc_topic_dist, p_lda, p_vectorizer, n_top_words=10)
sents_p = [ ' '.join(spacy_tokenizer(s)) for s in df.iloc[p_related_issue_index].sort_values(by=['time'])['body']]
print_issue("President Park Geun-hye Middle-East tour")
print_relatedissue_events(df, p_related_issue_index)
extract_entity_by_index(sents_p, df, p_related_issue_index)