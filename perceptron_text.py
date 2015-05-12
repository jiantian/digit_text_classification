
from __future__ import division
import sys, math, operator, collections, random
import numpy as np

class Perceptron:
	def __init__(self):
		# per-class weights
		self.weights = {}
		# bias (0 for no bias)
		self.bias = 0
		# maximum epoch
		self.max_epoch = 6
		# ordering of training examples
		self.order = "random"
		# train set
		self.trainset = []
		# train label
		self.trainlabel = []
		# testing label
		self.testlabel = []
		# predicted labels for the testing set
		self.predictLabel = []
		self.dictionary = {}

	def initial_weights(self):
		for i in range(8):
			# initialize weights as zeros
			size = len(self.dictionary)
			w = [0 for x in range(size)]
			self.weights[i] = w

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

	def learning_rate(self, epoch):
		return 1000 / (1000 + epoch)

	def train(self, training_file):
		self.getDict(training_file)
		self.initial_weights()
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

		for epoch in range(1, self.max_epoch):
			decay = self.learning_rate(epoch)
			correct_count = 0
			if self.order == "random":
				r = list(range(len(self.trainset)))
				random.shuffle(r)
				for i in r:
					line = lines[i]
					line = line.strip().split(' ')
					vector = self.trainset[i]
					label = int(line.pop(0))
					our_label = self.decision_rule(line, vector)
					if label != our_label:
						for j in range(len(line)):
							item = line[j]
							item = item.split(':')
							word = item[0]
							index = self.dictionary[word]
							self.weights[label][index] = self.weights[label][index] + decay * vector[index]
							self.weights[our_label][index] = self.weights[our_label][index] - decay * vector[index]
					else:
						correct_count += 1
			print "Epoch = ", epoch, ", Accuracy = %g" %(correct_count / len(self.trainset))

	def decision_rule(self, line, vector):
		c = {}
		for label in range(8):
			c[label] = 0
			for i in range(len(line)):
				item = line[i]
				item = item.split(':')
				word = item[0]
				if word in self.dictionary:
					index = self.dictionary[word]
					c[label] += self.weights[label][index] * vector[index]
		return max(c.iteritems(), key=operator.itemgetter(1))[0]

	"""do testing, print the results"""
	def test(self, test_file):
		lines = open(test_file, 'r').readlines()
		for line in lines:
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
			self.predictLabel.append(self.decision_rule(line, vector))
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
		print "Usage: python part2_1_text.py training_file test_file"
		sys.exit()

	perceptron = Perceptron()
	training_file = sys.argv[1]
	test_file = sys.argv[2]

	perceptron.train(training_file)
	perceptron.test(test_file)
