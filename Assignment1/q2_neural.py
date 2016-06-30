import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### forward propagation
    m, n = data.shape
    a0_m = np.concatenate((np.ones((m, 1)), data), 1)
    W1_m = np.concatenate((b1, W1), 0)
    W2_m = np.concatenate((b2, W2), 0)
    
    z1 = np.dot(a0_m, W1_m)
    a1 = sigmoid(z1)
    a1_m = np.concatenate((np.ones((m, 1)), a1), 1)
    z2 = np.dot(a1_m, W2_m)
    a2 = softmax(z2)
    # crossEntropy loss
    cost = -np.sum(labels * np.log(a2))
    
    ### backward propagation
    # softmax: e^xi / sum(e^x)
    # L: crossEntropy gradients
    delta2 = a2 - labels
    # L - 1: sigmoid gradient
    delta1 = np.dot(delta2, W2.T) * sigmoid_grad(a1)
    gradW2_m = np.dot(a1_m.T, delta2)
    gradW1_m = np.dot(a0_m.T, delta1) 

    gradW2 = gradW2_m[1:, :]
    gradb2 = gradW2_m[0, :]
    gradW1 = gradW1_m[1:, :]
    gradb1 = gradW1_m[0, :]

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad
    #return grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    # gradcheck_naive(func(param), param)
    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)
    #print forward_backward_prop(data, labels, params, dimensions)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
