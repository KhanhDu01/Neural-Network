import numpy as np
import json
import time
import random

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing the set of weights depending on how many inputs, and how many neurons we want in a layer
        self.weights = 0.00001 * np.random.randn(n_inputs, n_neurons)
        # Initializing the biases all to 0 depending on how many neurons we wanted in the layer
        self.biases = np.zeros((1, n_neurons))

        # Adam Optimizer
        self.opt_weights = AdamOptimizer(self.weights.shape)
        self.opt_biases = AdamOptimizer(self.biases.shape)

    def forward(self, inputs):
        if (len(np.array(inputs).shape) == 1):
            inputs = [inputs]
        self.inputs = np.array(inputs)
        # Creating the output
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweight = np.dot(np.array(self.inputs).T, np.array(dvalues))
        
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values to pass to previous layer
        self.dinputs = np.dot(dvalues, np.array(self.weights).T)
    
    def update_params(self):
        # Use Adam Optimizer to update weight and biases
        self.weights = self.opt_weights.update(self.weights, self.dweight)
        self.biases = self.opt_biases.update(self.biases, self.dbiases)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1- 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

class Activation_Softmax_Loss_Categoricalentropy:
    def forward(self, inputs, y_true):
        self.activation = Activation_Softmax()
        self.activation.forward(inputs)
        self.output = self.activation.output
        return Loss_CategoricalCrossentropy().calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) ==  2:
            y_true = np.argmax(y_true, axis = 1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class AdamOptimizer:
    def __init__(self, params_shape, learning_rate=0.03, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize first and second moment vectors for all parameters
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        self.t = 0  # timestep  

    def update(self, params, grads):
        self.t += 1

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)

        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Update parameters
        params_update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        params -= params_update

        return params


# -------------------------------File Extraction----------------------------------------------------------------------------

def read(filepaths):
    print("Starting file extraction")
    n_drawings = 0
    # Initialize a collection of different drawings
    collection = []
    labels = []
    for filepath in filepaths:
        n_drawings = 0
        print(f"Reading {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if n_drawings < 1:
                    if line.strip():
                        # The raw data of a drawing
                        drawing_json = json.loads(line.strip())["drawing"]
                        name = json.loads(line.strip())["word"]
                        drawing = []
                        for i in range(len(drawing_json)):
                            drawing.extend(
                                # zips a stroke in the format of (x, y, t)
                                list(zip(
                                    drawing_json[i][0],     #x
                                    drawing_json[i][1],     #y
                                    drawing_json[i][2]      #t
                                ))
                            )
                        
                        # Add collection of strokes as a drawing
                        
                        collection.append(drawing)
                        labels.append(name)
                    # Counts the number of drawings added
                    n_drawings += 1
        print(f"Successfully added {n_drawings} drawings")

    return collection, labels

dataset_files = [
    "data/The Great Wall of China.ndjson",
    "data/The Eiffel Tower.ndjson",
    "data/The Mona Lisa.ndjson",
    "data/full_raw_airplane.ndjson",

]

start = time.time()

collection, labels = read(dataset_files)


# Creating the different categories as an array
categories = []
with open ('data/categories.txt', 'r') as f:
    for line in f:
        categories.append(line[:-1])

# Categories as an index
categories_to_index = {category: i for i, category in enumerate(categories)}

# Creating a function for creating a one-hot vector for training
def one_hot(category):
    vec = np.zeros(len(categories), dtype=int)
    vec[categories_to_index[category]] = 1
    return vec

#Shuffle the data
combined = []
for drawing in collection:
    combined.append((drawing, labels[collection.index(drawing)]))
random.shuffle(combined)
collection, labels = zip(*combined)


# ------------------------Creating the model-----------------------------------------------

# !USE IF ALREADY HAVE A MODEL SAVED!
with open('main/model.json', 'r') as f:
    model = json.load(f)

dense1 = Layer_Dense(model['n_weights_layer1'], model['n_biases_layer1'])
dense1.weights = model['weights_layer1']
dense1.biases = model['biases_layer1']
dense1.opt_weights.m = np.array(model['opt_weights_m_layer1'])
dense1.opt_weights.v = np.array(model['opt_weights_v_layer1'])
dense1.opt_weights.t = np.array(model['opt_weights_t_layer1'])
dense1.opt_biases.m = np.array(model['opt_biases_m_layer1'])
dense1.opt_biases.v = np.array(model['opt_biases_v_layer1'])
dense1.opt_biases.t = np.array(model['opt_biases_t_layer1'])
activation1 = Activation_ReLU()

dense2 = Layer_Dense(model['n_weights_layer2'], model['n_biases_layer2'])
dense2.weights = model['weights_layer2']
dense2.biases = model['biases_layer2']
dense2.opt_weights.m = np.array(model['opt_weights_m_layer2'])
dense2.opt_weights.v = np.array(model['opt_weights_v_layer2'])
dense2.opt_weights.t = np.array(model['opt_weights_t_layer2'])
dense2.opt_biases.m = np.array(model['opt_biases_m_layer2'])
dense2.opt_biases.v = np.array(model['opt_biases_v_layer2'])
dense2.opt_biases.t = np.array(model['opt_biases_t_layer2'])
activation2 = Activation_ReLU()

dense3 = Layer_Dense(model['n_weights_layer3'], model['n_biases_layer3'])
dense3.weights = model['weights_layer3']
dense3.biases = model['biases_layer3']
dense3.opt_weights.m = np.array(model['opt_weights_m_layer3'])
dense3.opt_weights.v = np.array(model['opt_weights_v_layer3'])
dense3.opt_weights.t = np.array(model['opt_weights_t_layer3'])
dense3.opt_biases.m = np.array(model['opt_biases_m_layer3'])
dense3.opt_biases.v = np.array(model['opt_biases_v_layer3'])
dense3.opt_biases.t = np.array(model['opt_biases_t_layer3'])
loss_activation = Activation_Softmax_Loss_Categoricalentropy()
'''

# !USE TO CREATE A NEW MODEL!
dense1 = Layer_Dense(3, 64)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 64)
activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, len(categories))
loss_activation = Activation_Softmax_Loss_Categoricalentropy()
'''

# ------------------------------Training the model------------------------------------------------------------------------
n = 0
batch_size = 32
X_batch = []
y_batch = []

for drawing, label in zip(collection, labels):
    y = one_hot(label)
    for stroke in drawing:
        x_norm = stroke[0] / 256.0
        y_norm = stroke[1] / 256.0
        t_norm = stroke[2] / 10000.0 
        X = (x_norm, y_norm, t_norm)
        X_batch.append(X)
        y_batch.append(y)
        if len(X_batch) >= batch_size:
            X_batch_np = np.array(X_batch)
            y_batch_np = np.array(y_batch)
            # Forward pass
            dense1.forward(X_batch_np)
            activation1.forward(dense1.output)
            dense2.forward(activation1.output)
            activation2.forward(dense2.output)
            dense3.forward(activation2.output)
            
            # Loss and softmax
            loss = loss_activation.forward(dense3.output, y_batch_np)
            predictions = np.argmax(loss_activation.output, axis=1)
            targets = np.argmax(y_batch_np, axis=1)
            # Calculate accuracy
            accuracy = np.mean(predictions == targets)
            
            if n % 1000 == 0:
                print(f"Epoch {n}, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
            
            # Backward pass
            loss_activation.backward(loss_activation.output, y_batch_np)
            dense3.backward(loss_activation.dinputs)
            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)
            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            dense1.update_params()
            dense2.update_params()
            dense3.update_params()

            X_batch = []
            y_batch = []
            n += 1
    print(predictions)
    print(targets)
# If any leftover samples in the last batch
if X_batch:
    X_batch_np = np.array(X_batch)
    y_batch_np = np.array(y_batch)
    dense1.forward(X_batch_np)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y_batch_np)
    predictions = np.argmax(loss_activation.output, axis=1)
    targets = np.argmax(y_batch_np, axis=1)
    accuracy = np.mean(predictions == targets)
    print(f'Final Batch, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}')
    loss_activation.backward(loss_activation.output, y_batch_np)
    dense3.backward(loss_activation.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    dense1.update_params()
    dense2.update_params()
    dense3.update_params()
'''

# -------------------------------Saving the weight and biases-----------------------------------
print("Saving model parameters...")
# Save the weights and biases of each neuron layers
model_params = {
    'n_weights_layer1': 3,
    'weights_layer1': dense1.weights.tolist(),
    'opt_weights_m_layer1': dense1.opt_weights.m.tolist(),
    'opt_weights_v_layer1': dense1.opt_weights.v.tolist(),
    'opt_weights_t_layer1': int(dense1.opt_weights.t),
    'n_biases_layer1': 64,
    'biases_layer1': dense1.biases.tolist(),
    'opt_biases_m_layer1': dense1.opt_biases.m.tolist(),
    'opt_biases_v_layer1': dense1.opt_biases.v.tolist(),
    'opt_biases_t_layer1': int(dense1.opt_biases.t),
    'n_weights_layer2': 64,
    'weights_layer2': dense2.weights.tolist(),
    'opt_weights_m_layer2': dense2.opt_weights.m.tolist(),
    'opt_weights_v_layer2': dense2.opt_weights.v.tolist(),
    'opt_weights_t_layer2': int(dense2.opt_weights.t),
    'n_biases_layer2': 64,
    'biases_layer2': dense2.biases.tolist(),
    'opt_biases_m_layer2': dense2.opt_biases.m.tolist(),
    'opt_biases_v_layer2': dense2.opt_biases.v.tolist(),
    'opt_biases_t_layer2': int(dense2.opt_biases.t),
    'n_weights_layer3': 64,
    'weights_layer3': dense3.weights.tolist(),
    'opt_weights_m_layer3': dense3.opt_weights.m.tolist(),
    'opt_weights_v_layer3': dense3.opt_weights.v.tolist(),
    'opt_weights_t_layer3': int(dense3.opt_weights.t),
    'n_biases_layer3': 345,
    'biases_layer3': dense3.biases.tolist(),
    'opt_biases_m_layer3': dense3.opt_biases.m.tolist(),
    'opt_biases_v_layer3': dense3.opt_biases.v.tolist(),
    'opt_biases_t_layer3': int(dense3.opt_biases.t)
}

with open('main/model.json', 'w') as f:
    json.dump(model_params, f)
end = time.time()
print(f"Success. This took {(int(end - start))/60} minutes")'''