class Network(object):

    def __init__(self, sizes, file_name=None):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        #print(self.biases)
        #print(self.weights)
    def save(self, file_name='./data.npz'):
        np.savez(file_name, size=np.array(self.sizes), biases=np.array(self.biases), weights=np.array(self.weights))

    def load(self, file_name='./data.npz'):
        try:
            data = np.load(file_name)
        except:
            print('ERROR : Couldn\'t read the file.' )
            return
        
        try:
            self.sizes = list(data['size'])
            self.num_layers = len(self.sizes)
        except:
            print('ERROR : Couldn\'t read the variable sizes.' )
            return
        
        try:
            self.biases = list(data['biases'])
        except:
            print('ERROR : Couldn\'t read the variable biases.' )
            return
        
        try:
            self.weights = (data['weights'])
        except:
            print('ERROR : Couldn\'t read the variable weights.' )
            return
        
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def eta_handler(self, j):
        return self.t * (1 -(( (1.0459)**(j+1) )/10))
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        self.t = eta
        print('Before Training :' + str(self.evaluate(test_data)))
        if test_data: 
            n_test = len(test_data)
        #print(test_data)
        n = len(training_data)
        for j in range(epochs):
            eta = self.eta_handler(j)      #this is me, my own learning rate calculator
            random.shuffle(training_data)
            #print(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            #print(mini_batches)
            #print(self.evaluate(test_data))
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch "+str(j)+" : "+str(self.evaluate(test_data))+" / "+str(n_test)+"ETA : " + str(eta))
            else:
                print ("Epoch "+str(j)+" complete. ETA : " + str(eta))
                #self.evalv(testing_data,j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #print('hi')
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        #print(len(x))
        #print(x)
        #print(y)
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        #print(self.cost_derivative(activations[-1], y))
        #print(sigmoid_prime(zs[-1]))
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        #print(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    def evalv(self, test_data,j):
        test_results = [np.argmax(self.feedforward(x)) for x in test_data]
        file_1 = open('./sandbox/2.' + str(j) + '.' + 'submit.csv','w+')
        file_1.write('ImageId,Label')
        file_1.write('\n')
        for x, y in zip(test_results, range(1,28001)):
            file_1.write(str(y) + ',' + str(x))
            file_1.write('\n')
        
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
