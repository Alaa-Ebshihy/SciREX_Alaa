"""
contain functions for text preporcessing
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


def spacy_tokenizer(sentence, parser, stopwords, punctuations):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stopwords and word not in punctuations]
    mytokens = " ".join([i for i in mytokens])
    return mytokens


# TODO: possibility to add sci bert also
def vectorize_text(text, model='en_core_sci_lg'):
    if model == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=2 ** 12)
        return vectorizer.fit_transform(text)
    nlp = spacy.load(model)
    doc = nlp(text)
    return doc.vector
