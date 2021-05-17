import nltk
from nltk.tokenize import word_tokenize

# REPLACE HERE
file_txt = 's1.txt'

with open(file_txt, 'r') as file:
    
    data = file.read().replace('<s>', '').replace('</s>','').replace('\n', '')


unigrams = list(nltk.ngrams(data.split(), n=1))

bigrams = list(nltk.ngrams(data.split(), n=2))

print("Unigrams: ")
print(*map(' '.join, unigrams), sep=', ')

print("\nBigrams: ")
print(*map(' '.join, bigrams), sep=', ')

