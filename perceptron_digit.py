
from __future__ import division
import sys, math, operator, random
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
	def __init__(self):
		# per-class weights
		self.weights = {}
		self.initial_weights()
		# bias (0 for no bias)
		self.bias = 0
		# maximum epoch
		self.max_epoch = 15
		# ordering of training examples
		self.order = "random"
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

	def initial_weights(self):
		for i in range(10):
			# initialize weights as zeros
			w = [[0 for x in range(28)] for y in range(28)]
			self.weights[i] = w

	def learning_rate(self, epoch):
		return 1000 / (1000 + epoch)

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

	"""Training and updating weights"""
	def train(self):
		for epoch in range(1, self.max_epoch):
			decay = self.learning_rate(epoch)
			correct_count = 0
			if self.order == "random":
				r = list(range(len(self.trainset)))
				random.shuffle(r)
				for i in r:
					sample = self.trainset[i]
					label = self.trainlabel[i]
					our_label = self.decision_rule(sample)
					if label != our_label:
						for j in range(len(sample)):
							for k in range(len(sample)):
								self.weights[label][j][k] = self.weights[label][j][k] + decay * sample[j][k]
								self.weights[our_label][j][k] = self.weights[our_label][j][k] - decay * sample[j][k]
					else:
						correct_count += 1
			elif self.order == "fixed":
				for i in range(len(self.trainset)):
					sample = self.trainset[i]
					label = self.trainlabel[i]
					our_label = self.decision_rule(sample)
					if label != our_label:
						for j in range(len(sample)):
							for k in range(len(sample)):
								self.weights[label][j][k] = self.weights[label][j][k] + decay * sample[j][k]
								self.weights[our_label][j][k] = self.weights[our_label][j][k] - decay * sample[j][k]
					else:
						correct_count += 1
			else:
				sys.exit("No specified ordering!")
			print "Epoch = ", epoch, ", Accuracy = %g" %(correct_count / len(self.trainset)) 

	def decision_rule(self, sample):
		c = {}
		for label in range(10):
			c[label] = 0
			for i in range(len(sample)):
				for j in range(len(sample[0])):
					c[label] += self.weights[label][i][j] * sample[i][j]
		return max(c.iteritems(), key=operator.itemgetter(1))[0]

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
		for image in self.testset:
			self.predictLabel.append(self.decision_rule(image))
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
		print "Usage: python part2_1_digit.py training_file training_label test_file test_label"
		sys.exit()

	perceptron = Perceptron()
	training_file = sys.argv[1]
	training_label = sys.argv[2]
	test_file = sys.argv[3]
	test_label = sys.argv[4]
	perceptron.getTrainSet(training_file, training_label)
	perceptron.train()
	perceptron.getTestSet(test_file, test_label)
	perceptron.test()
