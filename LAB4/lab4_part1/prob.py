import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

new_file_txt = 'utf8_s1.txt'


with open(new_file_txt, 'r') as file:
    data = file.read().replace('<s>', '').replace('</s>','').replace('\n', '')

bigrams = list(nltk.ngrams(data.split(), n=2))

text = "Ele sofreu uma queda no primeiro jogo"

n = 2
train_data, padded_sents = padded_everygram_pipeline(n, bigrams)

model = MLE(n) 
model.fit(train_data, padded_sents)

print(model.vocab)

print(model.score('Ele'))
"""

model.counts['language'] # i.e. Count('language')
model.counts[['language']]['is'] # i.e. Count('is'|'language')
model.counts[['language', 'is']]['never'] # i.e. Count('never'|'language is')

model.score('is', 'language'.split())  # P('is'|'language')
model.score('never', 'language is'.split())  # P('never'|'language is')

"""