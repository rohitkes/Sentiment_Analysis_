import nltk 
from nltk.corpus import movie_reviews 
import random , pickle 
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier 
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC 
from nltk.classify import ClassifierI
from statistics import mode 
from collections import Counter 
import unicodedata 



# Create the document of word and the label as pos or neg 
documents = []

short_pos = open("short_reviews/positive.txt",'r').read()
short_neg = open("short_reviews/negative.txt",'r').read()


for pos_sent in short_pos.split("\n"):
	documents.append((pos_sent,"pos"))

for neg_sent in short_neg.split("\n"):
	documents.append((neg_sent,"neg"))

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

all_words = []
for w in short_pos_words:
	all_words.append(w.lower())

for w in short_neg_words:
	all_words.append(w.lower())



random.shuffle(documents)

# find the frequency of each word 
all_words = nltk.FreqDist(all_words)

#Take the top 3000 word coz rest will be useless
word_features = list(all_words.keys())[:5000]

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
	words = set(word_tokenize(document))
	features = {}
	for w in word_features:
		features[w] = (w in words)
	return features
#print((find_feature(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_feature(rev),cat) for (rev,cat) in documents]
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

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