
from __future__ import division
import sys, math, operator, random, time
import numpy as np
import numpy.linalg as la
import scipy as sp
import matplotlib.pyplot as plt

class KNN:
	def __init__(self):
		# parameter k in knn
		self.k = 4
		# training set
		self.trainset = []
		# training label
		self.trainlabel = []
		# testing set
		self.testset = []
		# testing label
		self.testlabel = []
		# predicted labels for the testing set
		self.predictLabel = []

	def distance(self, vector1, vector2):
		"""
		a1 = np.array(vector1).flatten()
		a2 = np.array(vector2).flatten()
		return sp.spatial.euclidean(a1,a2)
		return sp.spatial.cosine(a1,a2)
		return sp.spatial.hamming(a1,a2)
		return sp.spatial.cityblock(a1,a2)
		"""
		sum_sq = 0
		for i in range(28):
			for j in range(28):
				sum_sq += (vector1[i][j] - vector2[i][j])**2
		return math.sqrt(sum_sq)

	"""Format the input training data and labels"""
	def getTrainSet(self, training_file, training_label):
		# load training image file
		raw_images = open(training_file, 'r').readlines()
		# load training 
		raw_lables = open(training_label, 'r').readlines()
		#print len(lables), len(images)

		for i in range(len(raw_lables)):
			image_temp = raw_images[i*28:i*28+28]
			for j in range(28):
				image_temp[j] = list(image_temp[j].rstrip('\n'))
				for k in range(len(image_temp[j])):
					if image_temp[j][k] == ' ':
						image_temp[j][k] = 0
					else:
						image_temp[j][k] = 1
			self.trainset.append(image_temp)
			self.trainlabel.append(int(raw_lables[i]))

	"""
	Get k nearest neighbors
	@param test_point: a vector point
	"""
	def getNeighbors(self, test_point):
		distances = []
		for i in range(len(self.trainset)):
			train_point = self.trainset[i]
 			label = self.trainlabel[i]
			dist = self.distance(train_point, test_point)
			distances.append((label, dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for i in range(self.k):
			neighbors.append((distances[i][0], distances[i][1]))
		return neighbors

	"""
	Get the label of the test_point by kNN
	"""
	def getLabel(self, test_point):
		neighbors = self.getNeighbors(test_point)
		votes = {}
		for i in range(len(neighbors)):
			label = neighbors[i][0]
			votes[label] = 1 + votes.get(label, 0)
		sortedVotes = sorted(votes.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sortedVotes[0][0]

	"""Format the input testing data and labels"""
	def getTestSet(self, test_file, test_label):
		# load training image file
		raw_images = open(test_file, 'r').readlines()
		# load training 
		raw_lables = open(test_label, 'r').readlines()

		for i in range(len(raw_lables)):
			image_temp = raw_images[i*28:i*28+28]
			for j in range(28):
				image_temp[j] = list(image_temp[j].rstrip('\n'))
				for k in range(len(image_temp[j])):
					if image_temp[j][k] == ' ':
						image_temp[j][k] = 0
					else:
						image_temp[j][k] = 1
			self.testset.append(image_temp)
			self.testlabel.append(int(raw_lables[i]))

	"""get classification rate"""
	def classification_rate(self):
		self.class_rate = {}
		self.test_count = {}
		self.test_count_correct = {}
		for label in range(10):
			self.test_count[label] = 0
			self.test_count_correct[label] = 0
		overall = 0
		for i in range(len(self.testlabel)):
			self.test_count[self.testlabel[i]] += 1
			if self.predictLabel[i] == self.testlabel[i]:
				self.test_count_correct[self.testlabel[i]] += 1
				overall += 1
		for label in range(10):
			self.class_rate[label] = self.test_count_correct[label] / self.test_count[label]
		self.ovaerall_accuracy = overall / len(self.testlabel)

	"""get confusion matrix"""
	def confusion_matrix(self):
		matrix = np.zeros((10,10))
		for i in range(len(self.testlabel)):
			matrix[self.testlabel[i], self.predictLabel[i]] += 1
		for label in range(10):
			matrix[label,:] /= self.test_count[label]
		return matrix

	"""Do testing on the testing file"""
	def test(self):
		count = 1
		for image in self.testset:
			print "Do %i of %i testing samples" %(count,len(self.testset))
			count+=1
			self.predictLabel.append(self.getLabel(image))
		self.classification_rate()
		matrix = self.confusion_matrix()
		#print "Per class classification rate:"
		#print self.class_rate
		print "Overall accuracy is: %g" %(self.ovaerall_accuracy) 
		print "Confusion matrix is:"
		np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
		print(matrix[:,:]*100)

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print "Usage: python part2_2_digit.py training_file training_label test_file test_label"
		sys.exit()

	start_time = time.clock()
	knn = KNN()
	training_file = sys.argv[1]
	training_label = sys.argv[2]
	test_file = sys.argv[3]
	test_label = sys.argv[4]
	knn.getTrainSet(training_file, training_label)
	knn.getTestSet(test_file, test_label)
	knn.test()
	print "Time used: %g" %(time.clock() - start_time)
