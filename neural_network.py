import numpy as np
import random
class NeuralNetwork:
	def __init__(self, X, y):
		self.X = X
		self.y = y
		self.W1 = np.random.randn(3, 4) * 0.1
		self.b1 = np.array([0.2, 0.3,0.8, 0.9])
		self.W2 = np.random.randn(4, 1)* 0.1
		self.b2 = np.array([0.7])
	def ReLU(self, x):
			return np.maximum(0, x)
	def Sigmoid(self, x):
			return 1/(1  + np.exp(-x))
	def forward(self):
		self.Z1 = self.X @ self.W1 + self.b1
		self.A1 = self.ReLU(self.Z1)
		self.Z2 = self.A1 @ self.W2 + self.b2
		self.prediction = self.Sigmoid(self.Z2)
		self.Loss = np.mean((self.y - self.prediction)**2)
		return self.Loss
	def backward(self):
			
			dLoss_dprediction = -2 * (self.y - self.prediction)
			dprediction_dZ2 = self.prediction * (1- self.prediction)
			dZ2 = dLoss_dprediction*dprediction_dZ2
			self.dW2 = self.A1.T @ dZ2
			self.db2 = np.sum(dZ2, axis = 0)
			dA1 = dZ2 @ self.W2.T
			dZ1 = dA1 * (self.Z1 > 0)
			self.dW1 = self.X.T @ dZ1
			self.db1 = np.sum(dZ1, axis = 0)
	def update(self):
			learning_rate = 0.1
			self.W1 -= learning_rate * self.dW1
			self.b1 -= learning_rate * self.db1
			self.W2 -= learning_rate * self.dW2
			self.b2 -= learning_rate * self.db2
	def train(self):
		for epoch in range(1000):
			self.forward()
			self.backward()
			self.update()
			if epoch % 100 == 0:
				print("Epoch:", epoch)
				print("Loss:", self.Loss)

			
			
		
	
r = NeuralNetwork(np.random.randn(4,3)*0.1,np.array([[1.],[0.],[1.],[1.]]))

r.train()