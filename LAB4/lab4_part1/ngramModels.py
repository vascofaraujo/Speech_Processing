import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import codecs
from collections import Counter

def convert_UTF8(file):
    BLOCKSIZE = 1048576 # or some other, desired size in bytes

    sourceFileName = file
    targetFileName = "utf8_" + file
    with codecs.open(sourceFileName, "r", "iso-8859-1") as sourceFile:
        with codecs.open(targetFileName, "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
    return targetFileName

# REPLACE HERE
file_txt = 's1.txt'

new_file_txt = convert_UTF8(file_txt)

with open(new_file_txt, 'r') as file:
    data = file.read().replace('<s>', '').replace('</s>','').replace('\n', '')


"""
words = nltk.word_tokenize(data)
unigramList = []

frequenceUnigrams = nltk.FreqDist(words)
for unigram, index in frequenceUnigrams.items():
    unigramList.append(unigram)
"""

sents = nltk.sent_tokenize(data)
words = nltk.word_tokenize(data)

unigramList = []
bigramList = []

for sentence in sents:
    sequence = word_tokenize(sentence)
    for word in sequence:
        unigramList.append(word)
    bigramList.extend(list(nltk.ngrams(sequence,2)))
    

numberOfWords = len(unigramList)

frequenceUnigrams = nltk.FreqDist(unigramList)
frequenceBigrams = nltk.FreqDist(bigramList)

mostCommonUnigrams = frequenceUnigrams.most_common(10)
mostCommonBigrams = frequenceBigrams.most_common(10)

fileOut = "ngrams_" + file_txt

f = open(fileOut, "w")
f.write("Number of words in {}: {}\nMost common unigrams: {}\nMost common bigrams: {}".format(file_txt, numberOfWords, mostCommonUnigrams, mostCommonBigrams))
f.close()


