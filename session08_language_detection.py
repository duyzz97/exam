#!/usr/bin/env python
# coding: utf-8

# ## language detection
# source: https://thecleverprogrammer.com/2021/10/30/language-detection-with-machine-learning/
#

# https://scikit-learn.org/stable/user_guide.html
# sections 6.2.3 # text feature extraction "the bag of words"
# sections 3.1 Cross Validation
# section 1.9 Naive Bayes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
print(data.head())


# %%
data.isnull().sum()


# %%
data["language"].value_counts()


# %%
x = np.array(data["Text"])
y = np.array(data["language"])



cv = CountVectorizer()

X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.33,
                                                                    random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)


# %%
user = input("Enter a Text: ")
user_data = cv.transform([user]).toarray()
output = model.predict(user_data)
print(output)

# wordcloud for english sentences
# get all sentences in english and put them in a list.
eng_sentences = data[data["language"].values == output[0]].Text.to_list()
# prepare a countvectorizer to get unique features (words in this case) and their occurrence
# words between 5 to 20 characters:
cv1 = CountVectorizer(token_pattern=r'\b\w{5,20}\b')
# words consisting of 3 letters or less
cv2 = CountVectorizer(token_pattern=r'\b\w{1,3}\b')
# Combination of 3 to 6 words
cv3 = CountVectorizer(ngram_range=(3, 6))
x1 = cv1.fit_transform(eng_sentences)
ngrams1 = cv1.get_feature_names_out()

# get the frequency of the ngrams.
# The toarray() method returns the frequency of each ngram for each sentence in a list of list, each list corresponding to each sentence.
# By summing across lists we can obtain the frequency of occurence for each ngram across all sentences.
ngrams1_freq = sum(x1.toarray())

vocab1 = {}
i = 0
for k in ngrams1:
    vocab1[k] = ngrams1_freq[i]
    i += 1

x2 = cv2.fit_transform(eng_sentences)
ngrams2 = cv2.get_feature_names_out()

ngrams2_freq = sum(x2.toarray())

vocab2 = {}
i = 0
for k in ngrams2:
    vocab2[k] = ngrams2_freq[i]
    i += 1

x3 = cv3.fit_transform(eng_sentences)
ngrams3 = cv3.get_feature_names_out()

ngrams3_freq = sum(x3.toarray())

vocab3 = {}
i = 0
for k in ngrams3:
    vocab3[k] = ngrams3_freq[i]
    i += 1
# create the word cloud, using the dictionary created above
# Creating a circular mask
x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)
# initialize word cloud
wordcloud = WordCloud(mask=mask)
# generate word cloud with dictionary
wordcloud.generate_from_frequencies(vocab1)
# plot using matplotlib.pyplot
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

wordcloud.generate_from_frequencies(vocab2)
# plot using matplotlib.pyplot
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

wordcloud.generate_from_frequencies(vocab3)
# plot using matplotlib.pyplot
plt.imshow(wordcloud)
plt.axis('off')
plt.show()