from bs4 import BeautifulSoup
import pandas as pd
import re

def preprocess_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    txt = soup.get_text()
    txt = txt.replace('\n', ' ').replace('\t', ' ').replace('``', '"').replace("''", '"')
    # Remove URLs
    txt = ' '.join([word for word in txt.split() if not word.startswith('http')])
    return txt

def preprocess_csv(filename):
    df = pd.read_csv(filename)
    # df = df.iloc[:5]  # For testing purposes
    df['Document_HTML'] = df['Document_HTML'].apply(preprocess_html)
    return df

if __name__ == '__main__':
    df = preprocess_csv('documents_data.csv')
    df.to_csv('documents_data_preprocessed.csv', index=False)