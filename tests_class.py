import numpy as np 
class par():
	"""docstring for par"""
	def __init__(self, val):
		
		self.val = val
		self.grad = np.ones_like(val)
class test():
	"""docstring for test"""
	def __init__(self):
		
		self.W = par(np.arange(12).reshape(3,4))

	def fit(self):
		self.W.grad = self.W.val**2
t = test()
print(t.W.val.transpose())
t.fit()
print(t.W.grad)
