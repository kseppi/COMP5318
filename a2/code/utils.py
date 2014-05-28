import cPickle
import datetime
import sys
import scipy
import numpy as np
import pylab as pl
from sklearn.cross_validation import StratifiedKFold

# For feature selection
from sklearn.feature_selection import *
from sklearn.decomposition import *
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.svm import LinearSVC

# Latent Dirichlet ??
import gensim
from gensim.matutils import Sparse2Corpus
from gensim.matutils import corpus2csc
from gensim.models import LdaModel

# Make a simple progress bar so its easy to see how quickly data is being imported
class prog_bar:
	def __init__(self,size,bar_size = 40):
		# Initialise counters
		self.bar_size = bar_size
		self.count = 0
		self.percent = 0.0
		self.size = size
		self.start_time = datetime.datetime.now()
		self.str_length = 0
	# Print the status
	def update(self):
		# Update count, percent and time
		self.count += 1
		# Only update other stuff if its a full percentage higher
		if int(float(self.count)*100/self.size)>int(self.percent):
			self.percent = float(self.count)*100/self.size
		
		avg_time = float((datetime.datetime.now()-self.start_time).seconds)/(self.count) #Naive correction factor
		time_left = int(float((datetime.datetime.now()-self.start_time).seconds)/self.count*self.size*(100-self.percent)/100*1.1)
		bar_progress=int(self.percent/100*self.bar_size)	
		bar = "[" + "-"*bar_progress + " "*(self.bar_size-bar_progress) + "]"
		print_str = bar + " "  + str(self.percent) + "% (" + str(self.count)  + "/" + str(self.size) + ")"
		print_str = print_str + " Time estimate: " + str(datetime.timedelta(seconds=time_left))

			
		# Some print things to make the backspace work nicely
		old_len = self.str_length
		self.str_length = len(print_str)
		print_str = print_str + " "*(old_len-len(print_str))

		# Print out the data
		sys.stdout.write(print_str)
		sys.stdout.flush()
		sys.stdout.write("\b"*(len(print_str)))

		# If this is the end, print new line
		if self.percent == 100:
			sys.stdout.write("\n")
			print "Total time taken:",str(datetime.timedelta(seconds=(datetime.datetime.now()-self.start_time).seconds))

# Takes filname of name,vector file (where vector is in format n1,n2,n3...)
def load_X(filename):
	# Try to open it as if it was saved with cPickle
	names = None
	try:
		X = cPickle.load(open(filename))
	except cPickle.UnpicklingError as e:
		# X is not in sparse vector form yet because a text file was specified
		X,names = read_X(filename)
	return X,names

def load_y(filename):
	# Try to open it as if it was saved with cPickle
	try:
		y = cPickle.load(open(filename))
		y_names = []
	except cPickle.UnpicklingError as e:
		# X is not in sparse vector form yet because a text file was specified
		y,y_names = read_y(filename)
	return y,y_names
	

# Takes filename of name,label file
# returns dict of name -> label
def read_y(fname):
	fg = open(fname,"r")
	target = {}
	
	print "Reading targets from",fname
	lines = fg.readlines()
	pbar = prog_bar(len(lines))
	names = []
	for line in lines:
		pbar.update()
		name, klass = line.rstrip('\n').rstrip('\r').split(",")
		target[name] = klass
		names.append(name)

	fg.close()
	return target,names
# Takes filname of name,vector file (where vector is in format n1,n2,n3...)
# Returns dict of name -> vector
def read_X(fname):
	fv = open(fname,"r")
	vecs = {}

	# Set up some basic countint to see where we're up to
	count = 0
	# Set up the progress bar
	print "Reading vectors from",fname
	lines = fv.readlines()
	pbar = prog_bar(len(lines))

	# Read in the lines
	names = []
	for line in lines:
		count += 1
		# To show how far through we are
		pbar.update()	
		line = line.rstrip('\n').rstrip('\r')
		delim = line.find(",")
		#name,vec = line[:delim],[decimal.Decimal(x) for x in line[delim+1:].split(",")]
		name,vec = line[:delim],[float(x) for x in line[delim+1:].split(",")]
		vecs[name] = scipy.sparse.csc_matrix(np.array(vec))
		names.append(name)
	
	return vecs,names

# Takes the target and data file and returnx the numpy arrays
def make_vector(X_dict,y_dict,names):
	#shuffle(names) # in case the data is ordered. If it is it could cause problems with training
	print "Making X and y into vector format"
	X=[]
	y=[]
	pbar = prog_bar(len(names))
	for name in names:
		pbar.update()
		if X == []:
			if type(X_dict) != type(None):
				X = X_dict[name]
			if type(y_dict) != type(None):
				y = np.array([y_dict[name]])
		else:
			if type(X_dict) != type(None):
				X = scipy.sparse.vstack((X,X_dict[name]))
			if type(y_dict) != type(None):
				y = np.hstack((y,y_dict[name]))
	return X,y

class L1LinearSVC:
	def __init__(self, k):
		self.k = k
		self.y = None

	def fit(self, X, y):
		self.svc = LinearSVC(C=self.k, penalty="l1",
                                      dual=False, tol=1e-3)
		self.svc.fit(X, y)
		return self

	def transform(self, X):
		X = self.svc.transform(X)
		return X

class LatentDA:
	def __init__(self, k):
		self.k = k

	def fit(self, X, y):
		corpus = Sparse2Corpus(X.T)
		self.lda = LdaModel(corpus, num_topics=int(self.k))
		return self

	def transform(self, X):
		new_corpus = Sparse2Corpus(X.T)
		c_lda = self.lda[new_corpus]
		X = corpus2csc(c_lda).T
		return X

class tfidf_selector:
	def __init__(self,k):
		self.k = k
		self.idx = 0


	def fit(self, X, y):
		#X = X.asformat('csr')
		col_sum = X.sum(axis=0).A.squeeze()
		self.idx = np.argsort(col_sum)[-self.k:][::-1]
		return self

	def transform(self,X):
		X= X[:,self.idx]
		return X

class without_K:
	def __init__(self,k):
		if type(k) == type(1):
			self.k 	= [k]
		elif type(k) == type([]):
			self.k = k
		else:
			raise RuntimeError(str(k) + " not a valid k value. Must be integer or list of integers")
	def fit(self,X):
		return self

	def transform(self,X,y):
		for i in self.k:
			# if one feature has been removed, then there are now only size-1 features, so column i will be column i-1 (if i's are ordered)
			X = removecol(X,i-k.index(i))
		return X
#class remove_small:
#	def __init__(self,k):
#		self.threshold=k
#	def fit(self,X,y):
#		X = X.tocsr()
#		X = X.multiply(X >= self.threshold)
#		return X
#	def transform(self,X):
#		X = X.tocsr()
#		return X.multiply(X => self.threshold)

def reduce_features(X,y,method,k):
	if type(k) == type([]):
		k_list = k
		k = k[0]
	else:
		k_list = [k]

	if method == "PCA":
		model = PCA(n_components=k)
	elif method=="remove_small":
		model = remove_small(k)
	elif method == "RandomizedPCA":
		model = RandomizedPCA(n_components=k, whiten=True)
	elif method == "LDA":
		model = LDA(n_components=k)
	elif method=="RFE":
		svc=SVC(kernel="linear")
		model = RFE(svc,k,step=1)
	elif method=="tfidf":
		model = tfidf_selector(k)

	elif method=="L1LinearSVC":
		model = L1LinearSVC(k)

	elif method=="Without_k":
		model = without_K(k_list)

	elif method=="LatentDA":
		model = LatentDA(k)

	elif method == "Percentile":
		model = SelectPercentile(chi2, percentile=k)
	elif method == "KBest":
		model = SelectKBest(chi2, k=k)
	elif method == "SelectFpr":
		model = SelectFpr(chi2,alpha=k)
	elif method == "SelectFdr":
		model = SelectFdr(chi2,alpha=k)
	elif method == "SelectFwe":
		model = SelectFwe(chi2,alpha=k)
	elif method=="SparsePCA":
		model = SparsePCA(n_components=k)
	elif method=="TruncatedSVD":
		model = TruncatedSVD(k)
	elif method=="Fastica":
		model = fastica(X)
	else:
		print "PCA", "RandomizedPCA","LDA","RFE","tfidf","Without_k","Percentile","KBest","SelectFpr","SeletFdr","SelectFwe","SparsePCA","TrunkcatedSVD","Fastica"
		raise RuntimeError(method + " not an implemented feature reduction technique")
		
	print 1, X.shape	
	selector=model.fit(X,y)
	if isinstance(model, RandomizedPCA):
		print "PCA XPLIANED VARIANCE: ", np.sum(model.explained_variance_ratio_)
	print 2, X.shape
	return selector,selector.transform(X)
