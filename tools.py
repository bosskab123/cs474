import pandas as pd
import numpy as np
import os
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import string
import random
import unicodedata
from string import punctuation
from string import digits
from nltk.stem import WordNetLemmatizer
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_lg')

### Lemmatization tool
stemmer = WordNetLemmatizer()
### Change similar words to the same word
UN_WORD = "The United Nations"
US_WORD = "The United States"
NK_WORD = "North Korea"
SK_WORD = "South Korea"

similar_words = {
    # Change to "The United States"
    "U.S.": US_WORD,
    "US": US_WORD,
    "USA": US_WORD,
    "United States": US_WORD,
    "United States'": US_WORD,
    "The United States'": US_WORD,
    
    # Change to "North Korea"
    "NK": NK_WORD,
    "NK's": NK_WORD,
    "N. Korea": NK_WORD,
    "N. Korea's": NK_WORD,
    "North Korea's": NK_WORD,
    
    # Change to "South Korea"
    "SK": SK_WORD,
    "SK's": SK_WORD,
    "S. Korea": SK_WORD,
    "S. Korea's": SK_WORD,
    "South Korea's": SK_WORD,
    
    # Change to "The United Nations"
    "United Nations": UN_WORD,
    "United Nations'": UN_WORD,
    "The United Nations'": UN_WORD,
    "UN": UN_WORD,
}

### Transform function
def text_cleaning(s: str):
        
    def replace_strange_char(s: str):
        non_en_chars = {
            "’": "'",
            "‘": "'"
        }

        def remove_non_en_chars(txt):
            # remove non english characters
            txt = convert_latin_chars(txt)
            for char in non_en_chars.keys():
                txt = re.sub(char, non_en_chars[char], txt)
            txt = re.sub(r'[^\x00-\x7F]+', ' ', txt)
            return txt

        def convert_latin_chars(txt):
            # convert latin characters
            return ''.join(char for char in unicodedata.normalize('NFKD', txt) if unicodedata.category(char) != 'Mn')

        s = remove_non_en_chars(s)
        s = convert_latin_chars(s)
        return s
    s = replace_strange_char(s)
    for key,value in similar_words.items():
        s = re.sub(key, value, s)
    return s

def spacy_tokenizer(s: str):
    # Change similar terms to the same term
    new_str = text_cleaning(s)
    doc = nlp(new_str)
    # Group tokens
    matcher = Matcher(nlp.vocab)
    token_groupup_pattern = [
        [{"LOWER": "the"}, {"LOWER": "united"}, {"LOWER": "nations"}],
        [{"LOWER": "the"}, {"LOWER": "united"}, {"LOWER": "states"}],
        [{"LOWER": "north"}, {"LOWER": "korea"}],
        [{"LOWER": "south"}, {"LOWER": "korea"}],
    ]
    matcher.add("TermGroup",token_groupup_pattern)
    matches = matcher(doc)
    merge_doc = []
    for nid, start, end in matches:
        merge_doc.append((start,end))
    with doc.retokenize() as retokenizer:
        for i in range(len(merge_doc)-1,-1,-1):
            retokenizer.merge(doc[merge_doc[i][0]:merge_doc[i][1]])
        
    # Remove all stopword, punctuation, number
    tokens = [ token.lemma_.lower() for token in doc \
              if not token.is_stop and not token.is_punct and not token.like_num and token.lemma_.strip()!= '']
    return tokens

### Preprocess function for grouping similar topic
def preprocess_manual(s: str):
    # Change similar words to the same word
    new_str = text_cleaning(s)
    # Remove punctuation
    new_str = ''.join(ch if ch not in set(punctuation) else " " for ch in new_str)
    # Remove all single characters
    new_str = re.sub(r'\W', ' ', new_str)
    new_str = re.sub(r'\s+[a-zA-Z]\s+', ' ', new_str)
    new_str = re.sub(r'\^[a-zA-Z]\s+', ' ', new_str) 
    # Substituting multiple spaces with single space
    new_str = re.sub(r'\s+', ' ', new_str, flags=re.I)
    # Removing prefixed 'b' - when data is in bytes format
    new_str = re.sub(r'^b\s+', '', new_str)
    # Removing all numbers
    new_str = new_str.translate(str.maketrans('', '', digits))
    # Converting to Lowercase
    new_str = new_str.lower()
    # Lemmatization and remove stopwords
    new_str = new_str.split()
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [stemmer.lemmatize(word) for word in new_str if word not in stopwords]
    new_str = ' '.join(tokens)
    
    return new_str

### Make TF-IDF matrix
def tfidf_embed(documents, dimension=None):
    # documents: list of str
    # dim: integer
    embeddings_dict = {}
    tfidf_vectorizer = TfidfVectorizer(input='content', tokenizer=spacy_tokenizer)
    tfidf_vector = tfidf_vectorizer.fit_transform(documents)
    
    # Dimensionality Reduction
    if dimension is not None:
        svd_doc = TruncatedSVD(n_components=dimension, n_iter=5, random_state=42)
        tfidf_vector = svd_doc.fit_transform(tfidf_vector)
    return tfidf_vector

### Make GloVe matrix
glove_file = "../glove.42B.300d.txt"
def glove_word_vector():
    embeddings_dict = {}
    with open(glove_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

# Average sum of word vectors
def sentence_embed(sentence, word_vectors, dimension):
    sum_vector = np.zeros(dimension)
    for w in sentence.split():
        if w in word_vectors:
            sum_vector += word_vectors[w]
    return sum_vector/len(sentence)

# Make document vector
def document_embed(documents, embedding_technique='tfidf', dimension=None):
    if embedding_technique=='tfidf':
        doc_vector = tfidf_embed(documents, dimension)
    elif embedding_technique=='glove':
        word_vector = glove_word_vector()
        if dimension is None:
            dimension = 300
        doc_vector = [ sentence_embed(s, word_vector, dimension).tolist() for s in documents ]
    elif embedding_technique=='spacy':
        doc_vector = [doc.vector for doc in documents]
    
    return doc_vector

### Clustering 
def document_clustering(doc_vectors, clustering_method='kmeans'):
    if clustering_method=='kmeans':
        # Hyperparameters
        k_event = 10000
        k_issue = 6000
        
        # Clustering event
        kmeans_event = KMeans(n_clusters=k_event, random_state=69).fit(doc_vectors)
        # Represent each event by average sum of related news
        event_vectors = np.zeros((k_event, doc_vectors.shape[1]))
        for i in range(k_event):
            event_vectors[i] = sum(doc_vectors[kmeans_event.labels_ == i])
        
        # Clustering issue
        kmeans_issue = KMeans(n_clusters=k_issue, random_state=69).fit(event_vectors)
        # Represent each issue by average sum of related news
        issue_vectors = np.zeros((k_issue, doc_vectors.shape[1]))
        for i in range(k_issue):
            issue_vectors[i] = sum(event_vectors[kmeans_issue.labels_ == i])

        issue_labels = np.array([ kmeans_issue.labels_[kmeans_event.labels_[i]] for i in range(doc_vectors.shape[0]) ])
        
        return k_issue, k_event, issue_labels, kmeans_event.labels_
    
    elif clustering_method=='DBSCAN':
        
        # Hyperparameters
        doc_eps = 0.19
        doc_neighbors = 1
        event_eps = 0.50
        event_neighbors = 1
        '''
        Clustering using specific value
        '''
        # Clustering event
        db_event = DBSCAN(eps=doc_eps, min_samples=doc_neighbors).fit(doc_vectors)
        # Number of clusters in labels, ignoring noise if present.
        n_events_ = len(set(db_event.labels_)) - (1 if -1 in db_event.labels_ else 0)
        n_noise_ = list(db_event.labels_).count(-1)
        # Represent each event by average sum of related news
        event_labels = np.array(list(map(lambda x: n_events_ if x==-1 else x, db_event.labels_)))
        event_vectors = np.zeros((n_events_, doc_vectors.shape[1]))
        for i in range(n_events_+1):
            if np.sum(event_labels == i) != 0:
                event_vectors[i] = np.sum(doc_vectors[event_labels == i], axis=0)/np.sum(event_labels == i)

        # Clustering issue
        db_issue = DBSCAN(eps=event_eps, min_samples=event_neighbors).fit(event_vectors)
        # Number of clusters in labels, ignoring noise if present.
        n_issues_ = len(set(db_issue.labels_)) - (1 if -1 in db_issue.labels_ else 0)
        n_noise_ = list(db_issue.labels_).count(-1)
        # Represent each issue by average sum of related news
        issue_labels = np.array(list(map(lambda x: n_issues_ if x==-1 else x, db_issue.labels_)))
        issue_vectors = np.zeros((n_issues_, doc_vectors.shape[1]))
        for i in range(n_issues_+1):
            if np.sum(issue_labels == i) != 0:
                issue_vectors[i] = np.sum(event_vectors[issue_labels == i], axis=0)/np.sum(issue_labels == i)
    
        issue_labels = np.array([ issue_labels[event_labels[i]] for i in range(doc_vectors.shape[0]) ])
        
        return n_issues_, n_events_, issue_labels, event_labels
    
    elif clustering_method=='agglomerative':
        # Hyperparameters
        n_events = 10000
        n_issues = 6000
        
        # Clustering event
        agg_event = AgglomerativeClustering(distance_threshold=0, n_clusters=n_events).fit(doc_vectors)
        # Represent each event by average sum of related news
        event_vectors = np.zeros((n_events, doc_vectors.shape[1]))
        for i in range(n_events):
            event_vectors[i] = sum(doc_vectors[agg_event.labels_ == i])
        
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(agg_event, truncate_mode="level", p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()
        
        # Clustering issue
        agg_issue = AgglomerativeClustering(distance_threshold=0, n_clusters=n_issues).fit(event_vectors)
        # Represent each issue by average sum of related news
        issue_vectors = np.zeros((n_issues, doc_vectors.shape[1]))
        for i in range(n_issues):
            issue_vectors[i] = sum(event_vectors[agg_issue.labels_ == i])

        issue_labels = np.array([ agg_issue.labels_[agg_event.labels_[i]] for i in range(doc_vectors.shape[0]) ])
        
        return agg_issue, agg_event, issue_labels, agg_event.labels_
    
    else:
        assert("Doesn't support {}".format(clustering_method))

# Returns merged sentences given df and label
def merge_sents(df, label=100):
    return df[df.label == label]['agg_title_body']

# Print issue
def print_issue(issue):
    print("[ Issue ]\n")
    print(issue)
    print()

# Extract entities from sentences
def extract_entity_by_label(sents, df, label=100):
    print("[ Detailed Information (per event) ]\n")
    
    ent_per = []
    ent_org = []
    ent_loc = []
    
    event = df[df.label == label]['title'].iloc[::-1].values.tolist()
    
    for i in range(len(sents)):
        doc = nlp(sents[i])
        for ent in doc.ents:
            word = ent.text.title()
            if word in ent_per or word in ent_org or word in ent_loc:
                continue
            if ent.label_ == 'PERSON':
                ent_per.append(word)
            elif ent.label_ == 'ORG':
                ent_org.append(word)
            elif ent.label_ in ['GPE', 'LOC']:
                ent_loc.append(word)
        
        print("Event: ", event[i])
        print("- Person: ", ", ".join([i for i in ent_per]))
        print("- Organization: ", ", ".join([i for i in ent_org]))
        print("- Place: ", ", ".join([i for i in ent_loc]))
        print()

# Print on issue events
def print_onissue_events(df, label=100):
    event = df[df.label == label]['title'].iloc[::-1].values.tolist()
    print("[ On-Issue Events ]\n")
    print(" -> ".join([i for i in event]))
    print()

# Extract entities from sentences
def extract_entity_by_index(sents, df, index):
    print("[ Detailed Information (per event) ]\n")
    
    ent_per = []
    ent_org = []
    ent_loc = []
    
    event = df.iloc[index].sort_values(by=['time'])['title'].tolist()
    
    for i in reversed(range(len(sents))):
        doc = nlp(sents[i])
        for ent in doc.ents:
            word = ent.text.title()
            if word in ent_per or word in ent_org or word in ent_loc:
                continue
            if ent.label_ == 'PERSON':
                ent_per.append(word)
            elif ent.label_ == 'ORG':
                ent_org.append(word)
            elif ent.label_ in ['GPE', 'LOC']:
                ent_loc.append(word)
        
        print("Event: ", event[i])
        print("- Person: ", ", ".join([i for i in ent_per]))
        print("- Organization: ", ", ".join([i for i in ent_org]))
        print("- Place: ", ", ".join([i for i in ent_loc]))
        print()

# Print related issue event
def print_relatedissue_events(df, index):
    event = df.iloc[index].sort_values(by=['time'])['title'].tolist()
    print("[ Related-Issue Events ]\n")
    print(", ".join([i for i in event]))
    print()

# Get most related document to the topic
def most_related_docs(df, doc_topic_dist, topic_index, num_docs=5):
    if str(topic_index) in doc_topic_dist.columns:
        sorted_doc = doc_topic_dist.sort_values(by=[str(topic_index)], ascending=False)
    elif int(topic_index) in doc_topic_dist.columns:
        sorted_doc = doc_topic_dist.sort_values(by=[int(topic_index)], ascending=False)
    return df.iloc[sorted_doc[:num_docs].index]

#Select 5 topics that no word missile included in top 10 words
def related_issue_event(num_event, unwanted_word, df, doc_topic_dist, model, vectorizer, n_top_words):
    feature_names = vectorizer.get_feature_names_out()
    candidate_topic_idx = []
    for topic_idx, topic in enumerate(model.components_):
        a_set = set(unwanted_word)
        b_set = set([ feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        if not (a_set & b_set):
            candidate_topic_idx.append(topic_idx)
    candidate_event_idx = []

    for candidate in candidate_topic_idx:
        candidate_event_idx.append(most_related_docs(df, doc_topic_dist, candidate, num_docs=1).index.tolist()[0])
        if len(candidate_event_idx) > num_event:
            break
    return candidate_event_idx