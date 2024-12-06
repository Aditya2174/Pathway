from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from sumy.summarizers.lsa import LsaSummarizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import streamlit as st

import nltk

if 'chat_messages' not in st.session_state:
    nltk.download('punkt')

LANGUAGE = "english"

def summarize(context_docs, sentences_count=5):
    """
    Summarize the given text using LSA (Latent Semantic Analysis).
    """
    SENTENCES_COUNT = sentences_count
    # The text you want to summarize
    text = "\n\n".join(context_docs)

    # Create a parser for the input text
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    # Print the summary sentences
    summary = ""
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        #print(sentence)
        summary += str(sentence) + "\n"
    return summary

def clustered_rag_lsa(embedder, context_docs, num_clusters=5, sentences_count=5):
    """
    Cluster the context documents and summarize each cluster using LSA.
    """     
    if context_docs == []:
        return context_docs
    document_embeddings = embedder.encode(context_docs)

    # Set up KMeans clustering with KMeans++ initialization
    kmeans_plus_plus = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
    kmeans_plus_plus_labels = kmeans_plus_plus.fit_predict(document_embeddings)

    # Organize documents by cluster
    clustered_docs = {i: [] for i in range(num_clusters)}
    for doc, cluster in zip(context_docs, kmeans_plus_plus_labels):
        clustered_docs[cluster].append(doc)

    summaries = []

    for cluster, docs in clustered_docs.items():
        summary = summarize(docs, sentences_count)
        summaries.append(summary)

    return summaries