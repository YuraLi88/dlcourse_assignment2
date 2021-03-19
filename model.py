import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layer1 = FullyConnectedLayer(n_input,hidden_layer_size)
        self.ReLU1 = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size,n_output)
        # self.ReLU2 = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        self.layer1.W.grad *= 0
        self.layer2.W.grad *= 0
        self.layer1.B.grad *= 0
        self.layer2.B.grad *= 0
        X1 = self.layer1.forward(X)
        X1 = self.ReLU1.forward(X1)      
        X1 = self.layer2.forward(X1)
        # X1 = self.ReLU1.forward(X1)   
        loss,grad = softmax_with_cross_entropy(X1,y)

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        grad = self.layer2.backward(grad)
        grad = self.ReLU1.backward(grad)
        self.layer1.backward(grad)
        
        all_params = self.params()
        for param_key in all_params:
            param = all_params[param_key]
            loss_l2,grad_l2 = l2_regularization(param.value, self.reg)
            loss +=loss_l2
            param.grad += grad_l2
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!


        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        X1 = self.layer1.forward(X)
        X1 = self.ReLU1.forward(X1)      
        X1 = self.layer2.forward(X1)
        predictions = softmax(X1)
        pred = np.argmax(predictions,axis=1)
        return pred

    def params(self):
        result = {'W1':self.layer1.W,'B1':self.layer1.B,'W2':self.layer2.W,'B2':self.layer2.B, }

        # TODO Implement aggregating all of the params

        return result
