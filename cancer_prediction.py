import numpy as np
import random
class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		self.feature = X
		self.label = y


	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		ans = []
		for i in range(len(X)):
			score = []
			for j in range(len(self.feature)):
				score.append([self.distance(X[i], self.feature[j]), self.label[j]])
			score.sort(key=lambda x:x[0])
			count = 0
			res = []
			for l in range(self.k):
				res.append(score[l][1])
				if score[l][1] == 0:
					count += 1
			if count > self.k/2:
				ans.append(0)
			else:
				ans.append(1)
		ans = np.array(ans)
		return ans

class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def entropy(self, label):
		members, counts = np.unique(label, return_counts=True)
		entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(members))])
		return entropy

	def infoGrain(self, data, label):
		expectedInformation = self.entropy(label)
		members, counts = np.unique(data, return_counts=True)
		ans = []
		for j in range(len(members)):
			res = []
			for i in range(len(data)):
				if data[i] == members[j]:
					res.append(label[i])
			ans.append(res)
		neededInformation = np.sum((counts[i]/np.sum(counts)) * self.entropy(ans[i]) for i in range(len(members)))
		informationGain = expectedInformation - neededInformation
		return informationGain

	def highestScoreIndex(self, data, label):
		allInfoGain = []
		for i in range(len(data[0])):
			columns = [col[i] for col in data]
			allInfoGain.append(self.infoGrain(columns, label))
		highestScore = max(allInfoGain)
		highestIndex = allInfoGain.index(highestScore)
		return highestIndex

	def decision(self, label):
		map = {}
		for cate in label:
			if cate in map.keys():
				map[cate] += 1
			else:
				map[cate] = 1
		sortedMap = sorted(map.items(), key=lambda x:-x[1])
		res = sortedMap[0][0]
		return res

	def reorder(self, data, index, value, y):
		resD = []
		resY = []
		for d_i, y_i in zip(data, y):
			if d_i[index] == value:
				newChart = d_i[:index].tolist()
				newChart.extend(d_i[index+1:])
				resD.append(newChart)
				resY.append(y_i)
		resD = np.array(resD)
		resY = np.array(resY)
		return resD, resY

	def expandTree(self, categorical_data, y, allLabel):
		label = list(y)
		if len(categorical_data[0]) == 1:
			return self.decision(label)
		if np.equal(categorical_data[0], categorical_data).all():
			return self.decision(label)
		if label.count(label[0]) == len(label):
			return label[0]
		hIndex = self.highestScoreIndex(categorical_data, label)
		decisionTree = {allLabel[hIndex]: {}}
		highest = allLabel[hIndex]
		del(allLabel[hIndex])
		category = [data[hIndex] for data in categorical_data]
		for cate in set(category):
			newLabel = allLabel[:]
			resData, resY = self.reorder(categorical_data, hIndex, cate, label)
			decisionTree[highest][cate] = self.expandTree(resData, resY, newLabel)
		return decisionTree

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		allLabel = list(range(len(categorical_data[0])))
		self.tree = self.expandTree(categorical_data, y, allLabel)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		res = []
		for i in range(len(categorical_data)):
			res.append(self.getCategory(categorical_data[i], self.tree))
		res = np.array(res)
		return res

	def getCategory(self, data, tree):
		cate = random.randint(0, 1)
		for key in tree.keys():
			index = key
		newTree = tree[index]
		for newKey in newTree.keys():
			if data[index] == newKey:
				if newTree[newKey].__class__.__name__ != 'dict':
					cate = newTree[newKey]
				else:
					cate = self.getCategory(data, newTree[newKey])
		return cate


class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b


	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		s = steps
		while s > 0:
			for i in range(len(X)):
				sum = np.dot(self.w, X[i]) + self.b
				if sum < 0:
					sum = 0
				else:
					sum = 1
				for j in range(len(self.w)):
					self.w[j] += self.lr * (y[i]-sum) * X[i][j]
				s = s - 1

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		output = []
		for i in range(len(X)):
			sum = np.dot(self.w, X[i]) + self.b
			if sum < 0:
				sum = 0
			else:
				sum = 1
			output.append(sum)
		output = np.array(output)
		return output

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		self.inputValue = input
		output = np.dot(input, self.w) + self.b
		return output

	def backward(self, gradients):
		#Write backward pass here
		res = np.dot(gradients, self.w.T)
		self.w -= self.lr * gradients * self.inputValue.T
		self.b -= self.lr * gradients
		return res

class Sigmoid:

	def __init__(self):
		None

	def forward(self, input):
		#Write forward pass here
		self.i = input
		res = 1.0/(1.0 + np.exp(-input))
		return res

	def backward(self, gradients):
		#Write backward pass here
		der = gradients * (1 - 1.0/(1.0 + np.exp(-self.i))) * 1.0/(1.0 + np.exp(-self.i))
		return der