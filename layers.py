import numpy as np

def softmax(predictions):
        '''
        Computes probabilities from scores

        Arguments:
            predictions, np array, shape is either (N) or (batch_size, N) -
                classifier output

        Returns:
            probs, np array of the same shape as predictions - 
                probability for every class, 0..1
        '''
        max_items = np.max(predictions,axis=1)
        if len(predictions.shape)==1:
            predictions = predictions-max_items
            probs = np.exp(predictions)
            Z = np.sum(probs)
            probs /=Z
        else:
            predictions -= max_items[:,None]
            probs = np.exp(predictions)
            Z = np.sum(probs,axis=1)
            probs /=Z[:,None]
        
        return probs

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.sum(W**2)
    grad = 2*reg_strength*W
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    predictions = preds.copy()

    if len(predictions.shape)==1:
        predictions = predictions[None,:]

    max_items = np.max(predictions,axis=1)
    predictions -= max_items[:,None]

    Z= np.sum(np.exp(predictions), axis=1)
    dprediction = np.zeros(predictions.shape)
    if not isinstance(target_index,np.ndarray):
        loss = np.log(Z)-predictions[:,target_index]
        dprediction = np.exp(predictions)/Z
        dprediction[:,target_index] -= 1    
    else:
        loss = np.zeros(target_index.shape[0])
        it = np.nditer(Z, flags=['multi_index'], op_flags = ['readwrite'])
        while not it.finished:
            ix = it.multi_index[0]
            loss[ix] = np.log(Z[ix])-predictions[ix,target_index[ix]]
            dprediction[ix] = np.exp(predictions[ix])/Z[ix]
            dprediction[ix,target_index[ix]] -= 1
            it.iternext() 

    loss = np.mean(loss)
    dprediction = dprediction/dprediction.shape[0]
    return loss, dprediction



class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)
        self.velocity = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.mask = X<0
        return np.where(self.mask,0.,X)


    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = np.where(self.mask,0.,d_out)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        self.X = X
        return np.dot(X,self.W.value)+self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        dW = np.dot(self.X.transpose(),d_out)
        dB = d_out
        d_input = np.dot(d_out,self.W.value.transpose())
        self.W.grad += dW
        self.B.grad += np.sum(dB,axis=0)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
