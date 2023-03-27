import numpy as np
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

import re
import string

import sklearn.neural_network
import sklearn.pipeline
import sklearn.metrics
import sklearn.feature_extraction

tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words = nltk.corpus.stopwords.words('english')

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    punctuation = string.punctuation.replace(":","").replace("=", "").replace("(", "").replace(")", "").replace("!", "").replace("?", "").replace("'", "")
    text = "".join([c if c not in punctuation else " " for c in text])
    words = tokenizer.tokenize(text)
    words = [w for w in words if not w in stop_words]
    return ' '.join([lemmatizer.lemmatize(w) for w in words])

def main():
    df = pd.read_csv("fb_sentiment.csv")

    df["PreprocessedText"] = df["FBPost"].apply(preprocess_text)

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        df["PreprocessedText"], df["Label"], test_size=0.1, random_state=42)
    
    train_target = sklearn.preprocessing.LabelEncoder().fit_transform(train_target)
    test_target = sklearn.preprocessing.LabelEncoder().fit_transform(test_target)
    
    TfidfVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        lowercase=True, analyzer="word", ngram_range=(1, 3), sublinear_tf=True
    )

    estimator = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=150)

    model = sklearn.pipeline.Pipeline([("TfidfVectorizer", TfidfVectorizer), ("estimator", estimator)])

    model.fit(train_data, train_target)
    prediction = model.predict(test_data)
    accuracy = sklearn.metrics.accuracy_score(prediction, test_target)
    
    return accuracy * 100
    

if __name__ == "__main__":
    accuracy = main()
    print(accuracy)