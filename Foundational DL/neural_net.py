import math
import random

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors

class NeuralNetwork:
    """
    A simple neural network implementation for binary classification.  This class
    implements a neural network that can learn to classify data into two
    categories (binary classification).

    Args:
        layer_sizes (list):  A list specifying the number of neurons in each
            layer of the network.  For example, `[2, 3, 1]` would create a
            network with 2 input neurons, 3 neurons in a single hidden layer,
            and 1 output neuron.
        learning_rate (float):  Controls how much the network's weights are
            adjusted during each training step.  Smaller values lead to slower,
            more precise learning, while larger values can lead to faster, but
            potentially less stable learning. Defaults to 0.01.
        clip_value (float):  Limits the size of the weight adjustments during
            training. This helps prevent a problem called "exploding gradients,"
            where the adjustments become too large and the network fails to
            learn. Defaults to 5.0.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, clip_value=5.0):
        """
        Initializes the neural network.

        Sets up the network's structure, including the weights and biases
        connecting the neurons.  The weights and biases are the "learnable"
        parts of the network that are adjusted during training.

        Args:
            layer_sizes (list): Number of neurons at each layer [input, hidden..., output]
            learning_rate (float): Step size for weight updates.
            clip_value (float): Value to clip gradients at.
        """
        self.params = {}
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.clip_value = clip_value

        for i in range(1, len(layer_sizes)):
            input_size = layer_sizes[i-1]
            output_size = layer_sizes[i]

            xavier_stddev = math.sqrt(2 / (input_size + output_size))

            self.params[f'W{i}'] = [
                [random.gauss(0, xavier_stddev) for _ in range(output_size)]
                for _ in range(input_size)
            ]

            self.params[f'b{i}'] = [
                [random.uniform(-0.01, 0.01) for _ in range(output_size)]
            ]

    def _sigmoid(self, x):
        """The sigmoid activation function.

        This function takes a number as input and "squashes" it to a value
        between 0 and 1.  It's used to introduce non-linearity into the
        network, allowing it to learn more complex patterns.  It also
        provides a probability-like output.

        Args:
            x (float or list): The input value (or list of values).

        Returns:
            float or list: The sigmoid of the input (or list of sigmoids).
        """
        if isinstance(x, (list, tuple)):
            return [self._sigmoid(val) for val in x]

        x = max(min(x, 500), -500)
        return 1 / (1 + math.exp(-x))

    def _sigmoid_deriv(self, z):
        """
        The derivative of the sigmoid function.

        The derivative is used during backpropagation to calculate how much
        the weights and biases need to be adjusted.  It tells us the slope of
        the sigmoid function at a given point.

        Args:
            x (float or list): The input value (or list of values).  This should
                be the *output* of the sigmoid function, not the original input.

        Returns:
            float or list: The derivative of the sigmoid at the input (or list
                of derivatives).
        """
        if isinstance(z, (list, tuple)):
            return [self._sigmoid_deriv(val) for val in z]
        a = self._sigmoid(z)  # Calculate sigmoid(z)
        return a * (1 - a)
    
    def _matmul(self, A, B):
        """
        Matrix multiplication.

        This function performs matrix multiplication of two matrices, A and B.
        Matrix multiplication is a fundamental operation in neural networks,
        used to calculate the weighted sum of inputs at each neuron.

        Args:
            A (list of lists): The first matrix.
            B (list of lists): The second matrix.

        Returns:
            list of lists: The result of the matrix multiplication (A * B).

        Raises:
            ValueError: If the matrices are empty or have incompatible
                dimensions for multiplication.
        """
        if not A or not B or not A[0] or not B[0]:
            raise ValueError("Empty matrix in multiplication")
        return [[sum(a*b for a,b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    def _add_vectors(self, A, B):
        """
        Element-wise vector addition with broadcasting.

        This function adds two matrices (or vectors) together, element by
        element.  It also supports "broadcasting," which allows a smaller
        matrix (e.g., a bias vector) to be added to a larger matrix in a
        sensible way.

        Args:
            A (list of lists): The first matrix.
            B (list of lists): The second matrix.

        Returns:
            list of lists: The result of the element-wise addition (A + B).

        Raises:
            ValueError: If the matrices have incompatible dimensions for
                addition, even after considering broadcasting.
        """
        if len(B) == 1 and len(A) > 1:
            B = B * len(A)
        elif len(A) == 1 and len(B) > 1:
            A = A * len(B)

        if len(A) != len(B) or len(A[0]) != len(B[0]):
            raise ValueError(f"Matrix dimensions don't match: {len(A)}x{len(A[0])} vs {len(B)}x{len(B[0])}")
        return [[a + b for a, b in zip(A_row, B_row)] for A_row, B_row in zip(A, B)]

    def _init_visualization(self):
        """
        Initializes visualization components for plotting. Now creates
        subfigures for each layer's weights.
        """

        self.axs[0].set_title('Training Metrics')
        self.axs[0].set_xlabel('Epoch')
        self.axs[0].set_ylabel('Value')
        self.axs[0].legend(['Loss', 'Accuracy'], loc='upper right')
        plt.tight_layout()

    def _update_visualization(self, cache, epoch):
        """
        Updates visualization with the current network state, including
        weight matrices visualized as heatmaps.
        """

        num_layers = len(self.params) // 2
        if not hasattr(self, 'weight_figs'):
            self.weight_figs = []
            self.weight_axs = []
            for i in range(1, num_layers + 1):
                fig, ax = plt.subplots(figsize=(6, 4))
                self.weight_figs.append(fig)
                self.weight_axs.append(ax)
                ax.set_title(f'Weights W{i} (Epoch {epoch})')
                ax.set_xlabel('Neuron in Layer i')
                ax.set_ylabel('Neuron in Layer i-1')

        all_weights = []
        for key in self.params:
            if 'W' in key:
                all_weights.extend(np.ravel(self.params[key]))
        global_min = np.min(all_weights)
        global_max = np.max(all_weights)
        norm = matplotlib.colors.Normalize(vmin=global_min, vmax=global_max)

        for i in range(1, num_layers + 1):
            ax = self.weight_axs[i-1]
            ax.clear()
            weights = np.array(self.params[f'W{i}'])
            im = ax.imshow(weights, cmap='viridis', aspect='auto', norm=norm)
            ax.set_title(f'Weights W{i} (Epoch {epoch})')
            ax.set_xlabel('Neuron in Layer i')
            ax.set_ylabel('Neuron in Layer i-1')
            if not hasattr(self, f'cbar{i}'):
                cbar = self.weight_figs[i-1].colorbar(im, ax=ax)
                setattr(self, f'cbar{i}', cbar)
            else:
                getattr(self, f'cbar{i}').update_normal(im)

        self.axs[1].clear()
        for i in range(1, len(self.layer_sizes)):
            layer_activations = np.array(cache[f'A{i}'])
            if len(layer_activations) > 0: # prevent crash
                sample_activations = layer_activations[0]
                x = [i] * len(sample_activations)
                self.axs[1].scatter(x, sample_activations, alpha=0.5)
        self.axs[1].set_xticks(range(1, len(self.layer_sizes)))
        self.axs[1].set_title('Activation Patterns')
        self.axs[1].set_xlabel('Layer')
        self.axs[1].set_ylabel('Activation Value')

        self.axs[0].clear()
        self.axs[0].plot(self.loss_history, label='Loss', color='red')
        self.axs[0].plot(self.acc_history, label='Accuracy', color='green')
        self.axs[0].legend()
        self.axs[0].set_title('Training Metrics')

        plt.draw()
        plt.pause(0.001)

    def forward(self, X):
        """
        Performs a forward pass through the network.

        This function takes the input data (X) and propagates it through the
        network, layer by layer.  At each layer, it calculates the weighted sum
        of inputs, adds the bias, and applies the sigmoid activation function.
        The output of the final layer is the network's prediction.

        Args:
            X (list of lists): The input data.  Each inner list represents a
                single data sample.

        Returns:
            dict: A dictionary containing the intermediate values (activations
                and pre-activation values 'Z') calculated at each layer.  This
                is used later for backpropagation.

        Raises:
            ValueError: If the input data is empty.
        """
        if not X:
            raise ValueError("Empty input")

        cache = {'A0': X}
        num_layers = len(self.params) // 2

        for i in range(1, num_layers + 1):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            Z = self._add_vectors(self._matmul(cache[f'A{i-1}'], W), b)
            cache[f'Z{i}'] = Z
            cache[f'A{i}'] = [self._sigmoid(row) for row in Z]

        return cache

    def compute_loss(self, y_true, y_pred):
        """
        Computes the binary cross-entropy loss.

        This function calculates the difference between the network's
        predictions (y_pred) and the true labels (y_true).  Binary
        cross-entropy loss is commonly used for binary classification problems.
        It measures the dissimilarity between the predicted probabilities and
        the true labels.

        Args:
            y_true (list of lists): The true labels (0 or 1).
            y_pred (list of lists): The network's predictions (probabilities
                between 0 and 1).

        Returns:
            float: The average binary cross-entropy loss across all samples.
        """
        m = len(y_true)
        total = 0.0
        epsilon = 1e-15

        for true, pred in zip(y_true, y_pred):
            for t, p in zip(true, pred):
                p = max(min(p, 1 - epsilon), epsilon)
                total += t * math.log(p) + (1 - t) * math.log(1 - p)
        return -total / m

    def _transpose(self, matrix):
        """
        Calculates the transpose of a matrix.

        The transpose of a matrix swaps its rows and columns.  This operation
        is used in backpropagation.

        Args:
            matrix (list of lists): The matrix to transpose.

        Returns:
            list of lists: The transposed matrix.

        Raises:
            ValueError: If the input matrix is empty.
        """
        if not matrix or not matrix[0]:
            raise ValueError("Empty matrix in transpose")
        return list(map(list, zip(*matrix)))

    def backward(self, cache, y_true):
        """
        Performs backpropagation to calculate gradients.

        This function implements the backpropagation algorithm, which is used
        to calculate the gradients of the loss function with respect to the
        network's weights and biases.  These gradients indicate how much each
        weight and bias contributed to the error in the network's predictions.

        Args:
            cache (dict): The dictionary containing intermediate values from the
                forward pass.
            y_true (list of lists): The true labels.

        Returns:
            dict: A dictionary containing the gradients of the weights (dW) and
                biases (db) for each layer.
        """
        grads = {}
        m = len(y_true)  # Number of samples
        L = len(self.params)//2  # Number of layers

        # Output layer gradient
        A_L = cache[f'A{L}']
        dZ = [[a - t for a, t in zip(A_row, t_row)] for A_row, t_row in zip(A_L, y_true)]

        # Calculate gradients for output layer
        A_prev = cache[f'A{L-1}']
        grads[f'dW{L}'] = self._matmul(self._transpose(A_prev), dZ)
        grads[f'dW{L}'] = [[cell/m for cell in row] for row in grads[f'dW{L}']]
        grads[f'db{L}'] = [[sum(col)/m for col in zip(*dZ)]]

        # Backpropagate through hidden layers
        for l in reversed(range(1, L)):
            W_next = self.params[f'W{l+1}']
            dA = self._matmul(dZ, self._transpose(W_next))
            Z_l = cache[f'Z{l}']
            
            # Calculate dZ using the derivative of the sigmoid
            dZ = [[da * self._sigmoid_deriv(z) for da, z in zip(da_row, z_row)]
                   for da_row, z_row in zip(dA, Z_l)]

            A_prev = cache[f'A{l-1}']
            grads[f'dW{l}'] = self._matmul(self._transpose(A_prev), dZ)
            grads[f'dW{l}'] = [[cell/m for cell in row] for row in grads[f'dW{l}']]
            grads[f'db{l}'] = [[sum(col)/m for col in zip(*dZ)]]

            # Verify dimensions
            if len(grads[f'dW{l}']) != len(self.params[f'W{l}']):
                raise ValueError(f"Dimension mismatch in layer {l}")

        return grads

    def update_params(self, grads):
        """
        Updates the network's weights and biases using gradient descent.

        This function adjusts the weights and biases of the network based on
        the gradients calculated during backpropagation.  It uses the gradient
        descent algorithm, which moves the parameters in the direction that
        reduces the loss.

        Args:
            grads (dict): The dictionary containing the gradients of the
                weights and biases.
        """
        for key in self.params:
            param = self.params[key]
            grad = grads[f'd{key}']

            # Clip gradients to prevent exploding gradients
            grad = [[max(min(val, self.clip_value), -self.clip_value) 
                    for val in row] for row in grad]

            # Ensure dimensions match before updating
            if len(param) != len(grad):
                raise ValueError(f"Dimension mismatch for {key}: param shape {len(param)}x{len(param[0])}, "
                               f"grad shape {len(grad)}x{len(grad[0])}")

            # Update parameters using broadcasting for efficiency
            for i in range(len(param)):
                if len(param[i]) != len(grad[i]):
                    raise ValueError(f"Dimension mismatch in row {i} for {key}")
                for j in range(len(param[i])):
                    param[i][j] -= self.learning_rate * grad[i][j]

    def train(self, X, y, epochs=1000):
        """
        Trains the neural network.

        This function trains the network using the provided training data (X)
        and labels (y).  It repeatedly performs forward and backward passes,
        updating the network's parameters after each iteration (epoch).

        Args:
            X (list of lists): The training data.
            y (list): The training labels.
            epochs (int): The number of training iterations.  Defaults to 1000.

        Raises:
            ValueError: If the training data or labels are empty.
        """
        if not X or not y:
            raise ValueError("Training data cannot be empty")

        y_2d = [[val] for val in y]

        if len(X) != len(y):
            raise ValueError("Number of samples in X and y must match.")

        plt.ion()
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 4))
        self._init_visualization()
        self.loss_history = []
        self.acc_history = []

        for epoch in range(epochs):
            cache = self.forward(X)
            grads = self.backward(cache, y_2d)
            self.update_params(grads)

            if epoch % 100 == 0:
                loss = self.compute_loss(y_2d, cache[f'A{len(self.params)//2}'])
                acc = self.evaluate(X, y)
                print(f"Epoch {epoch} | Loss: {loss:.4f} | Train Acc: {acc:.2f}")

                self.loss_history.append(loss)
                self.acc_history.append(acc)
                self._update_visualization(cache, epoch)
                plt.draw()
                plt.pause(0.001)

        plt.ioff()
        plt.close(self.fig)
        for fig in self.weight_figs:
            plt.close(fig)
        return self.loss_history, self.acc_history

    def predict(self, X):
        """
        Makes predictions on new data.

        This function uses the trained network to make predictions on new,
        unseen data (X).  It performs a forward pass and then thresholds the
        output probabilities to produce binary predictions (0 or 1).

        Args:
            X (list of lists): The input data.

        Returns:
            list of lists: The network's predictions (0 or 1).
        """
        cache = self.forward(X)
        return [[1 if val > 0.5 else 0 for val in row] for row in cache[f'A{len(self.params)//2}']]

    def evaluate(self, X, y):
        """Evaluate accuracy, ensuring y is 2D"""
        predictions = self.predict(X)
        y_2d = [[val] for val in y]  # Convert y to 2D list here
        correct = sum(1 for p, true in zip(predictions, y_2d) if p[0] == true[0])
        return correct / len(y)

def generate_separable_data(num_samples=1000, num_features=5):
    """
    Generates linearly separable synthetic data for binary classification.

    This function creates a dataset where the two classes (0 and 1) can be
    perfectly separated by a hyperplane. This is useful for testing and
    demonstrating the neural network.

    Args:
        num_samples (int): The number of data samples to generate.
        num_features (int): The number of features (dimensions) for each sample.

    Returns:
        tuple: A tuple containing the input data (X) and the corresponding
               labels (y).  X is a list of lists, and y is a list.
    """
    X = []
    y = []
    for _ in range(num_samples):
        row = [random.gauss(0, 0.5) for _ in range(num_features)]
        if sum(row) > 0:
            y.append(1)
            row = [x + 0.5 for x in row]
        else:
            y.append(0)
            row = [x - 0.5 for x in row]
        X.append(row)
    return X, y

def normalize_data(X):
    """
    Normalizes the input data using z-score normalization.

    This function scales the features of the input data (X) so that each
    feature has a mean of 0 and a standard deviation of 1.  Normalization is
    important for neural networks to ensure that all features contribute
    equally to the learning process and to improve training stability.

    Args:
        X (list of lists): The input data.

    Returns:
        list of lists: The normalized input data.
    """
    means = [sum(col)/len(col) for col in zip(*X)]
    stds = [math.sqrt(sum((x-mean)**2 for x in col)/len(col)) for col, mean in zip(zip(*X), means)]
    epsilon = 1e-15
    return [
        [(x - mean) / (std + epsilon) for x, mean, std in zip(row, means, stds)]
        for row in X
    ]

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Splits the dataset into training and test sets.

    This function divides the data into two subsets: a training set used to
    train the network and a test set used to evaluate its performance on
    unseen data.  This is crucial for assessing how well the network
    generalizes to new data.

    Args:
        X (list of lists): The input data.
        y (list): The corresponding labels.
        test_size (float): The proportion of the data to use for the test set
            (e.g., 0.2 for 20%).  Defaults to 0.2.
        random_state (int):  Controls the random shuffling of the data.  Setting
            a specific value ensures that the split is reproducible.  Defaults
            to None.

    Returns:
        tuple: A tuple containing the training data (X_train), test data
            (X_test), training labels (y_train), and test labels (y_test).
    """
    random.seed(random_state)
    combined = list(zip(X, y))
    random.shuffle(combined)
    split_idx = int(len(combined) * (1 - test_size))
    train = combined[:split_idx]
    test = combined[split_idx:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(X_test), list(y_train), list(y_test)

def accuracy_score(y_true, y_pred):
    """
    Calculates the classification accuracy.

    This function measures the accuracy of the network's predictions by
    comparing them to the true labels.  Accuracy is the proportion of
    correctly classified samples.

    Args:
        y_true (list): The true labels.
        y_pred (list of lists): The network's predictions.

    Returns:
        float: The accuracy score (between 0 and 1).
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred[0]:
            correct += 1
    return correct / len(y_true)

if __name__ == "__main__":
    # Start with a simpler architecture for testing
    architectures = [
        [10, 16, 8, 1],  # Simpler architecture (input_size matches num_features)
    ]

    learning_rates = [0.1]  # Start with just one learning rate

    # Generate less data initially for faster testing
    X, y = generate_separable_data(num_samples=500, num_features=10)  # num_features matches input layer size
    X = normalize_data(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    default_architecture = architectures[0]
    default_lr = learning_rates[0]

    print(f"\n--- Training with Architecture: {default_architecture}, Learning Rate: {default_lr} ---")

    nn = NeuralNetwork(layer_sizes=default_architecture, learning_rate=default_lr)
    loss_history, acc_history = nn.train(X_train, y_train, epochs=1000)
    test_preds = nn.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Final Training Loss: {loss_history[-1]:.4f}")
    print(f"Final Training Accuracy: {acc_history[-1]:.4f}")