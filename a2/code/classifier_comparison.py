#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import RidgeClassifier
from utils import *
import scipy
import datetime
import cPickle
from utils import *
import argparse
import sys
from sklearn import cross_validation
import json
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import *

def get_classifiers(tests):
	global classifier_names
	global classifiers
	if len(tests) == 0:
		return classifiers
	my_classifiers = []
	for name in tests:
		if name in classifier_names:
			my_classifiers.append(classifiers[classifier_names.index(name)])
		else:
			raise RuntimeError(name + " is not a recognised classifier")
	return my_classifiers

def test_classifiers(test_name,X,y,tests=[],write_scores = "",cross_valid = 10,load=None,save = None,x_dense=False):
	print "Initiating classifiers"
	classifiers = get_classifiers(tests)

	# Split to train and test sets
	if not cross_valid:
			# SPlit to training and test set
			if type(load) ==type(None):
				print "Spliting to train and test sets"
				test_size=0.1
				X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
				if type(x_dense)!=type(False):
					print "Dense training data required. Extracting dense training data"
					start = datetime.datetime.now()
					X_train_dense, X_test_dense, y_train, y_test = train_test_split(x_dense, y, test_size=test_size)
					print "Done",datetime.datetime.now()-start
			else: #we're loading an already trained file so no need to get test data
				X_test,y_test = X,y
				X_train,y_train = None,None
				if type(x_dense)!=type(False):
					X_test = x_dense	

	# Write the heading of the test to file
	if write_scores:
		with open(write_scores,"a") as f:
			f.write("\nClassifier scoring for " + test_name + " (" + str(datetime.datetime.now()) + ")\n")
	with open("test_classifiers.log","a") as f:
		f.write("\nClassifier scoring for " + test_name + " (" + str(datetime.datetime.now()) + ")\n")
		

	# Loop through each clf
	for name, clf in zip(tests, classifiers):
		# Print out the heading and start the timer
		print "Testing",name
		time = datetime.datetime.now()

		# If we're doing cross validation then we don't need to split to training and test
		if cross_valid:
			print "Performing",cross_valid,"fold cross validation"
			try:
				scores = cross_validation.cross_val_score(clf,X,y,cv=cross_valid)
			except TypeError as e:
				if type(x_dense)!=type(False):
					print name,"requires a dense matrix. This could take a while"
					print "If cross validation is too slow, try restarting with --cross_valid 0"
					scores = cross_validation.cross_val_score(clf,x_dense,y,cv=cross_valid)
				else:
					raise RuntimeError(name + " requires a dense matrix. To test this, use the switch --dense")
		else:
			# If a filename was specified to load the clf:
			loaded = False
			if name in load.keys():
				# Get the filename and try to load it
				fname = load[name]
				print "Trying to load",fname
				try:
					clf = cPickle.load(open(fname))
					print "Loaded",fname
					loaded = True
				except Exception as pickle_e:
					print "Couldn't load",fname
					print pickle_e
					print "Setting to train the model instead"
			if not loaded:
					# We have to fit the data
					print "Training",
					try:
						clf.fit(X_train, y_train)
					except TypeError as e:
						if type(x_dense)!=type(False):
							clf.fit(X_train_dense, y_train)
						else:
							raise RuntimeError(name + " requires a dense matrix. To test this, use the switch --dense")


					# Do we have to save this clf?
					if name in save.keys():
						cPickle.dump(clf,open(save[name],"w"))	
			# Score the model
			print "Generating the score on the test set"
			try:
				scores = np.array([clf.score(X_test, y_test)])
			except TypeError as e:
				if type(x_dense)!=type(False):
					print name + " requires a dense matrix. This could take a while"
					scores = np.array([clf.score(X_test_dense, y_test)])
				else:
					raise RuntimeError(name + " requires a dense matrix. To test this, use the switch --dense")

		# Print out some information to stdout
		total_time = str(datetime.timedelta(seconds=(datetime.datetime.now()-time).seconds))
		print name + " Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() / 2) + "(" + total_time + ")"
		# Write the score to file
		if write_scores:
			with open(write_scores,"a") as f:
				f.write(name + " %0.4f (+/- %0.4f) " % (scores.mean(), scores.std() / 2) + "("+total_time+")")
			with open("test_classifiers.log","a") as f:
				f.write(name + " %0.4f (+/- %0.4f) " % (scores.mean(), scores.std() / 2) + "("+total_time+")" + "\n")

def get_dataset(name):
	if name == "kristy":
		X,y = read_matrix()
		X = reduce_features(X)
	elif test_name == "vanush":
		X = scipy.io.mmread('data_train_sparse_mm.txt')
		X = scipy.sparse.csr_matrix(X)
		X = reduce_features(X)
		y = np.loadtxt('data_train_label.txt', delimiter=',',dtype=str)
		y = y[:,1]
	else:
		raise RuntimeError("We don't have a dataset for" + name)
	return X,y
		

if __name__ == "__main__":
	# These are the classifiers we can test
	classifier_names = [
	    "Nearest_Neighbors", 
	    "SVM", 
	    "Linear_SVM", 
	    "RBF_SVM", 
	    "Decision_Tree",
	    "Random_Forest", 
	    "Naive_Bayes_Gaussian",
	    "Naive_Bayes_Multinomial",
	    "Naive_Bayes_Bernoulli",
	    "LDA", 
	    "Ridge",
	    "Dummy",
	    "Gradient_Boosting",
	    "Perceptron",
	    "Passive_Aggressive",
	    "SGD",
	    "Nearest_Centroid",
	    "Elastic_Net",
	    "Logistic_Regression"]
	classifiers = [
	    KNeighborsClassifier(3),
	    SVC(),
	    SVC(kernel="linear", C=0.025),
	    SVC(gamma=2, C=1),
	    DecisionTreeClassifier(max_depth=5),
	    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	    GaussianNB(),
	    MultinomialNB(alpha=.01),
	    BernoulliNB(alpha=.01),
	    LDA(),
	    RidgeClassifier(tol=1e-2,solver="lsqr"),
	    DummyClassifier(strategy='most_frequent',random_state=0),
	    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0),
	    Perceptron(n_iter=50),
	    PassiveAggressiveClassifier(n_iter=50),
	    SGDClassifier(alpha=.0001, n_iter=50,penalty="l1"),
	    NearestCentroid(),
	    SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet"),
	    LogisticRegression(C=1e5)]
	
	feature_reduction = ["tfidf","L1","tree","chi2"]
	
	# Get the arguments
	parser = argparse.ArgumentParser(description='Test a bunch of classifiers on a dataset.')
	datasets = ["kristy","vanush"]
	parser.add_argument("--dataset",help = "Type the dataset you want to load. The options are" + ",".join(datasets),required=True)
	parser.add_argument("--testname",nargs='?',help= "What do you want to call the test?",default="Test begining " + str(datetime.datetime.now()))
	parser.add_argument("--tests",nargs='+',help="Write what classifiers you want separated by space. Leave blank to test all. Options are: " + ",".join(classifier_names),type=str,default = classifier_names)
	parser.add_argument("--write_scores",nargs='?',help="Filename where you want to write the scores to",default = "")
	parser.add_argument("--cross_valid",nargs='?',help="The fold of cross validation. Leave blank to not use cross validation",type=int,default = 10)
	parser.add_argument("--load_clf",nargs='?',help="If there is a clf you have saved and want to load, then specifiy it as a dictionary of classifier name (specified in tests switch) map to filename. e.g. dict('{'Nearest_Neighbors':'nn.clf','Linear_SVM':'svm.clf'}')",type=json.loads,default = json.loads('{}'))
	parser.add_argument("--save_clf",nargs='?',help="If there is a clf you want saved, then specifiy it as a dictionary of classifier name (specified in tests switch) map to filename. e.g. dict('{'Nearest_Neighbors':'nn.clf','Linear_SVM:svm.clf'}')",type=json.loads,default = json.loads('{}'))
	parser.add_argument("--feature_reduction",help="The type of feature reduction you want. The options are: " + ",".join(feature_reduction) + ". Leave out for no feature reduction",default=None)
	parser.add_argument("--dense",action="store_true",default=False)

	args = parser.parse_args()

	print 
	print "Starting test with the following options:"
	print "Dataset:",args.dataset
	print "TestName:",args.testname
	print "Tests:",args.tests
	if args.dense:
		print "Using dense matrix where needed"
	if args.write_scores:
		print "Scores outputted to:",args.write_scores
	else:
		print "Scores not outputted to file"
	if args.cross_valid:
		print "Cross validation folds:",args.cross_valid
	else:
		print "Not performing cross validation"
	if len(args.load_clf.keys()) == 0:
		print "Not loading any clf's from file"
	else:
		print "Loading the following clfs", ",".join(args.load_clf.items())
	if len(args.save_clf.keys()) == 0:
		print "Not saving any clf's from file"
	else:
		print "Saving the following clfs", ",".join(args.save_clf.items())

	# Check there are no incorrect tests
	for test in args.tests:
		if test not in classifier_names:
			print "Classifiers must be in the list"
			print ",".join(classifier_names)
			raise RuntimeError(test + " not a valid classifier")
	print

	# Get X,y
	print "Loading Dataset"
	X,y = get_dataset(args.dataset)
	x_dense = args.dense
	if x_dense:
		print "Saving the dense version of the array"
		start = datetime.datetime.now()
		x_dense = X.toarray()
		print "Done",datetime.datetime.now()-start
	if args.feature_reduction not in feature_reduction:
		raise RuntimeError(args.feature_reduction + " not an implemented feature reduction technique. Choose from: " + ",".join(feature_reduction))
	if feature_reduction:
		print "Reducing Features..."
		start = datetime.datetime.now()
		X = reduce_features(X,y,feature_reduction) 
		print "Done",datetime.timedelta(seconds=(datetime.datetime.now()-start).seconds)

	test_classifiers(args.testname,X,y,tests=args.tests,write_scores = args.write_scores,cross_valid = args.cross_valid,load=args.load_clf,save = args.save_clf,x_dense=x_dense)
