
from __future__ import division
import sys, math, operator, collections, random, time
import numpy as np
import numpy.linalg as la
import scipy as sp

class KNN:
	def __init__(self):
		# parameter k in knn
		self.k = 5
		# train set
		self.trainset = []
		# train label
		self.trainlabel = []
		# testing label
		self.testlabel = []
		# predicted labels for the testing set
		self.predictLabel = []
		self.dictionary = {}

	"""get the word dictionary"""
	def getDict(self, training_file):
		lines = open(training_file, "r").readlines()
		index = 0
		for line in lines:
			line = line.strip().split(' ')
			label = int(line.pop(0))
			for item in line:
				item = item.split(':')
				word = item[0]
				if word not in self.dictionary:
					self.dictionary[word] = index
					index += 1

	def train(self, training_file):
		self.getDict(training_file)
		self.trainfile = []
		lines = open(training_file, "r").readlines()
		for line in lines:
			line = line.strip().split(' ')
			label = int(line.pop(0))
			self.trainlabel.append(label)
			vector = [0 for i in range(len(self.dictionary))]
			for item in line:
				item = item.split(':')
				word = item[0]
				count = int(item[1])
				if word in self.dictionary:
					index = self.dictionary[word]
					vector[index] = count
			self.trainset.append(vector)
			self.trainfile.append(line)

	def distance(self, vector1, vector2):
		"""
		a1 = np.array(vector1)
		a2 = np.array(vector2)
		return sp.spatial.euclidean(a1,a2)
		return sp.spatial.cosine(a1,a2)
		return sp.spatial.hamming(a1,a2)
		return sp.spatial.cityblock(a1,a2)
		"""
		sum_sq = 0
		for i in range(len(vector1)):
			sum_sq += (vector1[i] - vector2[i])**2
		return math.sqrt(sum_sq)

	"""
	Get k nearest neighbors
	@param test_point: a vector point
	"""
	def getNeighbors(self, testline, test_point):
		distances = []
		testindex = set()
		for item in testline:
			item = item.split(':')
			word = item[0]
			count = int(item[1])
			if word in self.dictionary:
				index = self.dictionary[word]
				testindex.add(index)
		for i in range(len(self.trainset)):
			train_point = self.trainset[i]
			label = self.trainlabel[i]
			train_line = self.trainfile[i]
			trainindex = set()
			for item in train_line:
				item = item.split(':')
				word = item[0]
				count = int(item[1])
				if word in self.dictionary:
					index = self.dictionary[word]
					trainindex.add(index)
			"""
			allindex = testindex.union(trainindex)
			sum_sq = 0
			for index in allindex:
				if index in testindex and index in trainindex:
					sum_sq += (test_point[index] - train_point[index])**2
				elif index in testindex and index not in trainindex:
					sum_sq += test_point[index]**2
				elif index not in testindex and index in trainindex:
					sum_sq += train_point[index]**2
			dist = math.sqrt(sum_sq)
			#dist = self.distance(train_point, test_point)
			"""
			allindex = testindex.intersection(trainindex)
			res = 0
			for index in allindex:
				res += test_point[index] * train_point[index]
			sum1 = 0
			for index in testindex:
				sum1 += test_point[index]**2
			sum2 = 0
			for index in trainindex:
				sum2 += train_point[index]**2
			dist = 1 - res / math.sqrt(sum1) / math.sqrt(sum2)
			distances.append((label, dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for i in range(self.k):
			neighbors.append((distances[i][0], distances[i][1]))
		return neighbors

	"""
	Get the label of the test_point by kNN
	"""
	def getLabel(self, testline, test_point):
		neighbors = self.getNeighbors(testline, test_point)
		votes = {}
		for i in range(len(neighbors)):
			label = neighbors[i][0]
			votes[label] = 1 + votes.get(label, 0)
		sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

	"""do testing, print the results"""
	def test(self, test_file):
		lines = open(test_file, 'r').readlines()
		ct = 1
		for line in lines:
			#print "Do %i of %i testing samples" %(ct,len(lines))
			ct += 1
			line = line.strip().split(' ')
			label = int(line.pop(0))
			self.testlabel.append(label)
			vector = [0 for i in range(len(self.dictionary))]
			for item in line:
				item = item.split(':')
				word = item[0]
				count = int(item[1])
				if word in self.dictionary:
					index = self.dictionary[word]
					vector[index] = count
			self.predictLabel.append(self.getLabel(line, vector))
		self.classification_rate()
		print "Overall accuracy is: %g" %(self.overall_accuracy)
		matrix = self.confusion_matrix()
		print "Confusion matrix is:"
		np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
		print(matrix[:,:]*100)

	"""get the per-class and overall accuracy"""
	def classification_rate(self):
		self.class_rate = {}
		self.test_count = {}
		self.test_count_correct = {}
		for label in range(8):
			self.test_count[label] = 0
			self.test_count_correct[label] = 0
		overall = 0
		for i in range(len(self.testlabel)):
			self.test_count[self.testlabel[i]] += 1
			if self.predictLabel[i] == self.testlabel[i]:
				self.test_count_correct[self.testlabel[i]] += 1
				overall += 1
		for label in range(8):
			self.class_rate[label] = self.test_count_correct[label] / self.test_count[label]
		self.overall_accuracy = overall / len(self.testlabel)

	"""get confusion matrix"""
	def confusion_matrix(self):
		dim = 8
		matrix = np.zeros((dim,dim))
		for i in range(len(self.testlabel)):
			matrix[self.testlabel[i], self.predictLabel[i]] += 1
		for label in range(8):
			matrix[label,:] /= self.test_count[label]
		return matrix

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print "Usage: python part2_2_text.py training_file test_file"
		sys.exit()

	start_time = time.clock()
	knn = KNN()
	training_file = sys.argv[1]
	test_file = sys.argv[2]

	knn.train(training_file)
	knn.test(test_file)
	print "Time used: %g" %(time.clock() - start_time)
