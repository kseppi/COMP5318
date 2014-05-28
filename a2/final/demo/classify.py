# Some testing tools
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import *

# Classifiers
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import *
from sklearn.dummy import *
from sklearn.cluster import *

# utilities
from utils import *
import cPickle
import argparse

if __name__ == "__main__":
	classifier_names = [
	    "Nearest_Neighbors",
	    "Nearest_Centroid",
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
	    "Gradient_Boosting",
	    "Perceptron",
	    "Passive_Aggressive",
	    "SGD",
	    "Nearest_Centroid",
	    "Elastic_Net",
	    "Logistic_Regression",
	    "Dummy"]
	classifiers = [
	    KNeighborsClassifier(15),
	    NearestCentroid(),
	    SVC(),
	    LinearSVC(C=0.8),
	    SVC(gamma=2, C=1),
	    DecisionTreeClassifier(max_depth=5),
	    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	    GaussianNB(),
	    MultinomialNB(alpha=.01),
	    BernoulliNB(alpha=.01),
	    LDA(),
	    RidgeClassifier(tol=1e-2,solver="lsqr"),
	    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0),
	    Perceptron(n_iter=50),
	    PassiveAggressiveClassifier(n_iter=50),
	    SGDClassifier(alpha=.0001, n_iter=50,penalty="l1"),
	    NearestCentroid(),
	    SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet"),
	    LogisticRegression(C=1e5),
	    DummyClassifier(strategy="stratified")]
	feature_reductions = ["tfidf","Without_k","chi2","KBest"]
	
	# Define the commandline switches
	parser = argparse.ArgumentParser(description='Run the classifier')
	parser.add_argument("--train_data",help="The vectors you want to train on",default=None)
	parser.add_argument("--train_data_save",help="Save the vector file for faster loading next time",default=None)
	parser.add_argument("--train_labels",help="The labels for each training vector",default=None)
	parser.add_argument("--train_labels_save",help="Saves labels for each training vector for faster loading next time",default=None)
	parser.add_argument("--test_data",help="The vectors you want to test on. Leave out if you want to split the training data into a test and train set",default = None)
	parser.add_argument("--test_data_save",help="Save the vector file for faster loading next time",default=None)
	parser.add_argument("--test_labels",help="The labels for each test vector. Leave out if you want to split the training data into a test and train set",default = None)
	parser.add_argument("--test_labels_save",help="Saves labels for each test vector for faster loading next time",default=None)
	parser.add_argument("--test_names_save",help="Saves order of names for each test vector for faster loading next time",default=None)
	parser.add_argument("--test_size",help="If you want to split the train set into a test set and train set but no perform cross evaluation, what proportion (as a decimal between 0 and 1) do you want the test set to be?",type=float,default=0.0)
	parser.add_argument("--save_predictions",help="File you want to save the predictions to",default =None)	
	clf_data = ["predict","predict_proba","score"]
	parser.add_argument("--clf_data",help="Print out clf data. Options are" + ",".join(clf_data),nargs="*",default=[])
	metrics_data = ["classification_report","f1_score","precision_score","recall_score","accuracy_score"]
	parser.add_argument("--metrics_data",help="Print out metrics data. Options are" + ",".join(metrics_data),nargs="*",default=[])
	parser.add_argument("--clf_file",help="If you have a pre-trained clf, where is it located?",default = None)
	parser.add_argument("--clf_name",help="What is the name of the clf you want to use? Note: if you specify a clf file then don't specify it's name. Options are:" + ",".join(classifier_names),default=None)
	parser.add_argument("--save_clf",help="This will train the clf with all of the train data and save it to the file name specified",default=None)
	parser.add_argument("--feature_reduction",help="The type of feature reduction you want to use. You may need to specify a k value with --k. Options are" + ",".join(feature_reductions),default=None)
	parser.add_argument("--feature_selector_file",help="If you have already trained a feautre selector and you want to load it",default=None)
	parser.add_argument("--save_selector",help="If you want to save a feature selector, specify its name",default=None)
	parser.add_argument("--k",nargs="*",type=float,help="The k value for the associated feature reduction technique",default = 0)
	parser.add_argument("--dense",help="Use the dense matrix instead",action='store_true',default=False)
	parser.add_argument("--v",help="Verbose. Print what's happening",action="store_true",default=False)
	parser.add_argument("--output_labels",help="Print line separated labels of test prediction",action="store_true",default=False)
	args = parser.parse_args()

	# Test some basic rules
	if args.clf_name and args.clf_name not in classifier_names:
		print parser.print_help()
		print ",".join(classifier_names)
		raise RuntimeError(args.clf_name + "not implemented")
	for d in args.clf_data:
		if d not in clf_data:
			print parser.print_help()
			print args.clf_data
			print clf_data
			raise RuntimeError(d + " not an implemented clf data metric")
	for d in args.metrics_data:
		if d not in metrics_data:
			print parser.print_help()
			print metrics_data
			raise RuntimeError(d + " not an implemented data metric")
	#if args.train_data and not args.train_labels or args.train_labels and not args.train_data:
	#	print parser.print_help()
	#	raise RuntimeError("If you specify train data (--train_data), you also need to specify their labels (--train_labels) and vice versa")
	if not args.test_data and not args.test_size and not args.save_clf:
		print "Warning: no test data, test size or save_clf switch is on"
		#print parser.print_help()
		#raise RuntimeError("There is no data to test on (--test_data or --test_size) and you don't want to save the clf (--save_clf). So running this would produce nothing.")
	if args.clf_file and args.clf_name:
		print parser.print_help()
		raise RuntimeError("You can't specify a clf file (--clf_file) and also a clf name (--clf_name)")
	if not args.train_data and not args.test_data:
		print parser.print_help()
		raise RuntimeError("You must specify at least test data (--test_data) or train data(--train_data) otherwise there is no data to run")
	if args.feature_selector_file and args.feature_reduction:
		print parser.print_help()
		raise RuntimeError("You can't specify both a feature selector to load (--feature_selection_file and a type of feauture reduction to perform (--args.feature_reduction)")
	if bool(args.train_data_save) != bool(args.train_labels_save):
		raise RuntimeError("If saving train data you must specify both --train_data_save and --train_labels_save")
	#if bool(args.test_data_save) != bool(args.test_labels_save):
	#	raise RuntimeError("If saving train data you must specify both --test_data_save and --test_labels_save")
	if args.test_size and args.test_data:
		print parser.print_help()
		raise RuntimeError("You can't specify both the test proportion and a test data set")
		
	
	# Load training data
	if args.train_data:
		if args.v:
			print "Loading training data from file"
		X_train,train_names = load_X(args.train_data)
		y_train,train_names = load_y(args.train_labels)
		# If the load function only read the files, we need to convert them to a proper matrix
		if type(X_train) == type({}) or type(y_train) == type({}):
			X_train,y_train = make_vector(X_train,y_train,train_names)
		if args.test_size:
			if args.v:
				print "Splitting into test-train set"
				print X_train.size,y_train.size
			X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)
			test_names = [i for i in range(len(y_test[0]))]
			train_names = [i for i in range(len(y_train[0]))]


		# Save train data
		if args.train_data_save:
			cPickle.dump(X_train,open(args.train_data_save,"w"))
		if args.train_labels_save:
			cPickle.dump(y_train,open(args.train_labels_save,"w"))
		
	else:
		X_train = None
		y_train = None
		

	# Load test data
	if args.test_data:
		if args.v:
			print "Loading test data from file"
		X_test,test_names = load_X(args.test_data)
		if args.test_labels:
			y_test,test_names = load_y(args.test_labels)
		else:
			y_test = None
		# If the load function only read the files, we need to convert them to a proper matrix
		if type(X_test) == type({}) or type(y_test) == type({}):
			X_test,y_test = make_vector(X_test,y_test,test_names)
			if y_test == []:
				y_test = None
	if test_names == None:
		test_names = [i for i in range(X_test.size)]
	# Save test data
	if args.test_data_save:
		cPickle.dump(X_test,open(args.test_data_save,"w"))
	if args.test_labels_save:
		cPickle.dump(y_test,open(args.test_labels_save,"w"))

	# Does the user want dense data?
	if args.dense:
		if args.v:
			print "Converting to dense data"
		X_train = X_train.toarray()
		X_test = X_test.toarray()

	# Reduce features
	selector = None
	if args.feature_reduction:
		if args.v:
			print "Training selector"
			print "Reducing features for train data"
			print "X_train shape:",X_train.shape
		selector,X_train = reduce_features(X_train,y_train,args.feature_reduction,args.k)
		if args.v:
			print "New X_train shape",X_train.shape
		if args.save_selector:
			cPickle.dump(selector,open(args.save_selector,"w"))
	if args.feature_selector_file:
		# Loading feature selector from file and applying to data
		if args.v:
			print "Loading selector from file"
		selector = cPickle.load(open('args.feature_selector_file'))
		if type(X_train) != type(None):
			if args.v:
				print "Reducing features for train data"
				print "X_train shape",X_train.shape
			X_train = selector.transform(X_train)
			if args.v:
				print "New X_train shape",X_train.shape
	if selector and type(X_test) != type(None):
		if args.v:
			print "Reducing features for test data"
			print "X_test shape",X_test.shape
		X_test = selector.transform(X_test)
		if args.v:
			print "New X_test shape",X_test.shape

	# Get clf
	if args.clf_file:
		if args.v:
			print "Loading clf from file"
		clf = cPickle.load(open(args.clf_file))
	elif args.clf_name:
		if args.v:
			print "Making clf"
		clf = classifiers[classifier_names.index(args.clf_name)]
	
		# Train the clf
		if type(X_train) != type(None):
			if args.v:
				print "Training clf"
			clf.fit(X_train,y_train)
			if args.save_clf:
				cPickle.dump(clf,open(args.save_clf,"w"))
	# Print out the labels
	if args.output_labels:
		X_pred = clf.predict(X_test)
		for p in X_pred:
			print p

	# Test the clf
	if type(X_test) != type(None):
		y_pred = clf.predict(X_test)
		if args.save_predictions:
			with open(args.save_predictions,"w") as f:
				for pred in y_pred:
					f.write(pred + "\n")
		for output_data in args.clf_data:
			if output_data == "predict":
				print "Predictions:"
				for name,pred in zip(test_names,y_pred):
					print name,pred
			elif output_data == "score" and type(y_test) != type(None):
				print "Score:",clf.score(X_test,y_test)
			elif output_data == "predict_proba":
				print "Probabilities"
				probs = predict_proba(X_test)
				for name,probs in zip(test_names,probs):
					print name, probs
			else:
				print "Don't know how to output",output_data
				print "Chose from",",".join(clf_data)
		# Metrics can only be outputted if gold data exists
		print
		if type(y_test) != type(None):
			target_names = list(set(y_test))
			for output_data in args.metrics_data:
				if output_data == "classification_report":
					print classification_report(y_test,y_pred,target_names=target_names)
				elif output_data == "f1_score":
					print f1_score(y_test,y_pred)
				elif output_data == "precision_score":
					print precision_score(y_test,y_pred)
				elif output_data == "recall_score":
					print recall_score(y_test,y_pred)
				elif output_data == "accuracy_score":
					print accuracy_sore(y_test,y_pred)
				else:
					print "Dont know how to output",output_data
				
				


			
		
