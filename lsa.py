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

import nltk
nltk.download('punkt')

LANGUAGE = "english"

def summarize(context_docs, sentences_count=10):

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

def clustered_rag_lsa(question, context_docs):

    embedder = SentenceTransformer("all-MiniLM-L6-v2") 
    document_embeddings = embedder.encode(context_docs)

    # Set the number of clusters
    num_clusters = 20

    # Step 2: Set up KMeans clustering with KMeans++ initialization
    kmeans_plus_plus = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
    kmeans_plus_plus_labels = kmeans_plus_plus.fit_predict(document_embeddings)

    # Organize documents by cluster
    clustered_docs = {i: [] for i in range(num_clusters)}
    for doc, cluster in zip(context_docs, kmeans_plus_plus_labels):
        clustered_docs[cluster].append(doc)

    summaries = []

    for cluster, docs in clustered_docs.items():
        summary = summarize(docs, sentences_count=5)
        summaries.append(summary)

    prompt = f"""
    Given the following summary of information retrieved from multiple documents, answer the question accurately and concisely.

    Use the information in the summary to directly address the question,
    Highlight relevant insights or facts that support the answer,
    If applicable, provide a brief, clear explanation to clarify the answer.
    Summaries: {summaries}

    Question: {question}

    Provide a concise answer based solely on the summaries above.
    """

    response = llm.generate_content(prompt)

    return response.text