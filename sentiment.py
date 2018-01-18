'''
from nltk.tokenize import sent_tokenize, word_tokenize 

print(sent_tokenize(text))


# Lecture 2 stopwords 
from nltk.corpus import stopwords 

print(stopwords.words("english"))

# Lecture 3 stemming 
riding and ride are same 
from nltk.stem import PorterStemmer 
ps = PorterStemmer()
print(ps.stem(word_tokenized_word))

#Lecture 4 part of speech tagging 
from nltk.corpus import state_union 
from nltk.tokenizer import PunktSentenceTokenizer 
import nltk 

custom_sent_tokenizer = PunktSentenceTokenizer(train_file)
tokenizer = custom_sent_tokenizer.tokenize(test_file)
words = word_tokenize(tokenizer[i])
nltk.pos_tag(words)

# Lecture 5 Chunking 

chunkGram = regular_expression 
chunkParser  = nltk.RegexpParser(chunkGram)
chunked = chunkParser.parse(tagged_word)

#Lecture 6 Chinking 

\





#Lecture 7 Named Entity Recognition 
namedEntity = nltk.ne_chunk(tagged)


#Lecture 8 lemmatizing // similiar to stemming but end result is real word 
# But in case of stem word may be wrong 

from nltk.stem import WordNetLemmatizer 
print(lemmatizer.lemmatize(word,pos="a"))  //pos =noun by default

# Lecture 9 corpus 

# Lecture 10 WordNet 
from nltk.corpus import wordnet 
sysns = WordNet.synsets(word)
rpint(sysns[0].lemmas()[0].name())  return synms 
sysns[0].definition()
print(sysns[0].lemmas()[0].antonyms)

'''

#Lecture 10 Text Classification sentiment analysis 
import nltk 
from nltk.corpus import movie_reviews 
import random , pickle 
from nltk.classify.scikitlearn import SklearnClassifier 
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC 
from nltk.classify import ClassifierI
from statistics import mode 
from collections import Counter 


# Create the document of word and the label as pos or neg 
documents = []
i=0
for category in movie_reviews.categories():
	for fileid in movie_reviews.fileids(category):
		documents.append((list(movie_reviews.words(fileid)),category))
		
		


random.shuffle(documents)


# get all the word 

all_words = []

for w in movie_reviews.words():
	all_words.append(w.lower())

# find the frequency of each word 
all_words = nltk.FreqDist(all_words)

#Take the top 3000 word coz rest will be useless
word_features = list(all_words.keys())[:3000]

#document is one file 

class VoteClassifier(ClassifierI):
	def __init__(self,*classifiers):
		self._classifiers = classifiers

	def classify(self,features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return Counter(votes).most_common(1)[0][0]

	def confidence(self,features):	
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(Counter(votes).most_common(1)[0][0])
		conf = choice_votes/float(len(votes))
		return conf 

def find_feature(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features
#print((find_feature(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_feature(rev),cat) for (rev,cat) in documents]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes accuracy percent:",nltk.classify.accuracy(classifier,testing_set)*100)


with open("naivebayesclassifier.pickle","wb") as f:
	pickle.dump(classifier,f)

classifier_f = open("naivebayesclassifier.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classification accuracy: ",nltk.classify.accuracy(MNB_classifier,testing_set)*100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classification accuracy: ",nltk.classify.accuracy(BernoulliNB_classifier,testing_set)*100)



LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classification accuracy: ",nltk.classify.accuracy(LogisticRegression_classifier,testing_set)*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classification accuracy: ",nltk.classify.accuracy(SGDClassifier_classifier,testing_set)*100)


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classification accuracy: ",nltk.classify.accuracy(LinearSVC_classifier,testing_set))



voted_classfier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, 
	SGDClassifier_classifier, LinearSVC_classifier)

print("Voted Classifier accuracy percentage ",nltk.classify.accuracy(voted_classfier,testing_set)*100)
print("Classification:",voted_classfier.classify(testing_set[0][0]), "Confidence:",voted_classfier.confidence(testing_set[0][0])*100)
print("Classification:",voted_classfier.classify(testing_set[1][0]), "Confidence:",voted_classfier.confidence(testing_set[1][0])*100)
print("Classification:",voted_classfier.classify(testing_set[2][0]), "Confidence:",voted_classfier.confidence(testing_set[2][0])*100)
print("Classification:",voted_classfier.classify(testing_set[3][0]), "Confidence:",voted_classfier.confidence(testing_set[3][0])*100)
print("Classification:",voted_classfier.classify(testing_set[4][0]), "Confidence:",voted_classfier.confidence(testing_set[4][0])*100)
print("Classification:",voted_classfier.classify(testing_set[5][0]), "Confidence:",voted_classfier.confidence(testing_set[5][0])*100)