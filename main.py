import pandas as pd
import numpy as np
import re
from nltk import sent_tokenize
from nltk.corpus import stopwords
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def print_df(df, lines):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(df.head(lines))
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def open_csv():
    try:
        return pd.read_csv('fake_reviews_processed.csv')
    except FileNotFoundError:
        print("Processed input doesn't exist, opening unprocessed instead.")

    df = pd.read_csv('fake_reviews_dataset.csv')
    df = clean_data(df)
    df = process_data(df)
    return df


def clean_data(df):
    print("Cleaning data...")
    # Remove special characters
    # df['text_'] = df['text_'].apply(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
    df['text_'] = df['text_'].apply(lambda x: re.sub('[\t\n\r\v\f]+', ' ', x))

    # Remove punctuation
    # df['text_'] = df['text_'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))

    # Remove stop words
    # stop = stopwords.words('english')
    # df['text_'] = df['text_'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

    # Lowercase
    df['text_'] = df['text_'].apply(lambda x: x.lower())

    # Filter empty reviews
    df['text_'].replace(['', ' '], np.nan, inplace=True)
    df.dropna(subset=['text_'], inplace=True)

    return df


def process_data(df):
    print("Processing data...")
    df['word_count'] = df['text_'].apply(lambda x: len(x.split()))
    df['average_word_size'] = df['text_'].apply(lambda x: sum(len(x) for x in x.split()) / len(x.split()))
    df['sentences'] = df['text_'].apply(lambda x: len(sent_tokenize(x)))
    df['average_sentence_length'] = df['text_'].apply(lambda x: len(x) / len(sent_tokenize(x)))
    # print_df(df, 100)
    df.to_csv('fake_reviews_processed.csv', index=False)
    return df


def train_and_predict(df):
    print("Training...")
    X = df.drop(['category', 'label', 'text_'], axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    score = accuracy_score(y_test, predictions)

    print(score)
    print(classification_report(predictions, y_test))

    print(X.columns)
    while True:
        x = input("Enter review: ")
        if len(x) == 0:
            continue
        rating = input("Enter rating: ")
        word_count = len(x.split())
        average_word_size = sum(len(x) for x in x.split()) / len(x.split())
        sentences = len(x) / len(sent_tokenize(x))
        average_sentence_length = len(x) / len(sent_tokenize(x))
        i = pd.DataFrame([[rating, word_count, average_word_size, sentences, average_sentence_length]],
                         columns=['rating', 'word_count', 'average_word_size', 'sentences', 'average_sentence_length'])
        print(model.predict(i))


def main():
    df = open_csv()
    train_and_predict(df)


if __name__ == '__main__':
    main()
