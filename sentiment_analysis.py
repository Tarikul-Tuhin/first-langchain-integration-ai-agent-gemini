import pandas as pd
import nltk

data = pd.read_csv('Restaurant_Reviews.tsv',sep='\t')

# print(data.head())

# data['char_count'] = data['Review'].apply(len)
# print(data.head())

# data['word_count'] = data['Review'].apply(lambda x:len(str(x).split()))
# print(data.head())


nltk.download('punkt')