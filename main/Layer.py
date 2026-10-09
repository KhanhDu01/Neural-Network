from pathlib import Path

import numpy as np
import json
import time
import random
from PIL import Image, ImageDraw

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing the set of weights depending on how many inputs, and how many neurons we want in a layer
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1.0 / n_inputs)
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
    def __init__(self, params_shape, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

        # GRADIENT CLIPPING - prevents explosion
        grad_norm = np.linalg.norm(grads)
        max_norm = 0.85  # Clip gradients to maximum norm of 1.0
        
        if grad_norm > max_norm:
            grads = grads * (max_norm / grad_norm)

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

class QuickDrawPreprocessor:
    def __init__(self, image_size = 28):
        self.image_size = image_size

    def load_dataset(self, files, max_samples_per_category=5000):
        X, y = [], []

        for filename in files:
            category = Path(filename).name
            category = category.removeprefix("full_simplified_")
            category = category.removesuffix(".ndjson")

            print(f"Loading {category}...")

            with open(filename, "r", encoding="utf-8") as f:
                count = 0

                for line in f:
                    if count >= max_samples_per_category:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line in {filename}: {line.strip()}")
                        continue

                    if not record.get("recognized", True):
                        continue

                    X.append(self.load_drawing(record))
                    y.append(category)
                    count += 1

            print(f"Loaded {count} samples for {category}")

        return np.array(X, dtype=np.float32), np.array(y)

    # Load data from filepaths
    def load_drawing(self, file, max_samples=10000):
        canvas = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(canvas)

        for stroke in file["drawing"]:
            xs = stroke[0]
            ys = stroke[1]

            points = list(zip(xs, ys))

            if len(points) == 1:
                x, y = points[0]
                draw.point((x, y), fill=255)
            elif len(points) > 1:
                draw.line(points, fill=255, width=5)

        canvas = canvas.resize((self.image_size, self.image_size))
        return np.asarray(canvas, dtype=np.float32).flatten() / 255
def load_categories(filepath="data/categories.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# -------------------------------File Extraction----------------------------------------------------------------------------

if __name__ == "__main__":

    dataset_files = [
        "data/full_simplified_The Eiffel Tower.ndjson",
        "data/full_simplified_The Mona Lisa.ndjson",
        "data/full_simplified_The Great Wall of China.ndjson",

    ]

    preprocessor = QuickDrawPreprocessor()


    start = time.time()

    print("Loading and preprocessing the dataset...")
    # Load and preprocess the dataset
    collection, labels = preprocessor.load_dataset(dataset_files)
    X = collection
    categories = load_categories("data/categories.txt")

    # Only keep categories represented in the loaded dataset
    unique_labels = sorted(set(labels.tolist()))

    # Validate that every dataset category exists in categories.txt
    missing = set(unique_labels) - set(categories)

    if missing:
        raise ValueError(
            f"These dataset categories are missing from categories.txt: {missing}"
        )

    # Use the order defined by categories.txt
    unique_labels = [c for c in categories if c in unique_labels]

    label_to_index = {
        category: i for i, category in enumerate(unique_labels)
    }

    y = np.array(
        [label_to_index[label] for label in labels],
        dtype=np.int64
    )

    y_onehot = np.eye(len(unique_labels), dtype=np.float32)[y]

    print("Categories:", unique_labels)
    print("Number of categories:", len(unique_labels))
    print("X shape:", collection.shape)
    print("y shape:", y.shape)
    print("One-hot shape:", y_onehot.shape)


    # ------------------------Creating the model-----------------------------------------------
    '''
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
    num_classes = len(unique_labels)

    dense1 = Layer_Dense(784, 128)
    activation1 = Activation_ReLU()

    dense2 = Layer_Dense(128, 128)
    activation2 = Activation_ReLU()

    dense3 = Layer_Dense(128, num_classes)

    loss_activation = Activation_Softmax_Loss_Categoricalentropy()

    # ------------------------------Training the model------------------------------------------------------------------------


    epochs = 10
    batch_size = 64

    for epoch in range(epochs):
        start_time = time.time()

        # Shuffle the drawings and labels together
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y_onehot[indices]

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(X), batch_size):
            X_batch = X_shuffled[start:start + batch_size]
            y_batch = y_shuffled[start:start + batch_size]
            batch_n = len(X_batch)

            # ---------------- Forward pass ----------------
            dense1.forward(X_batch)
            activation1.forward(dense1.output)

            dense2.forward(activation1.output)
            activation2.forward(dense2.output)

            dense3.forward(activation2.output)

            # Softmax + categorical cross-entropy
            loss = loss_activation.forward(
                dense3.output, y_batch
            )

            # Measure predictions
            predictions = np.argmax(
                loss_activation.output, axis=1
            )
            targets = np.argmax(y_batch, axis=1)

            total_loss += loss * batch_n
            total_correct += np.sum(predictions == targets)
            total_samples += batch_n

            # ---------------- Backward pass ----------------
            loss_activation.backward(
                loss_activation.output, y_batch
            )

            dense3.backward(loss_activation.dinputs)

            activation2.backward(dense3.dinputs)
            dense2.backward(activation2.dinputs)

            activation1.backward(dense2.dinputs)
            dense1.backward(activation1.dinputs)

            # ---------------- Update parameters ----------------
            dense1.update_params()
            dense2.update_params()
            dense3.update_params()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Time: {elapsed:.1f}s"
        )

    # -------------------------------Saving the weight and biases-----------------------------------
    print("Saving model parameters...")
    # Save the weights and biases of each neuron layers
    model_params = {
        'n_weights_layer1': 784,
        'weights_layer1': dense1.weights.tolist(),
        'opt_weights_m_layer1': dense1.opt_weights.m.tolist(),
        'opt_weights_v_layer1': dense1.opt_weights.v.tolist(),
        'opt_weights_t_layer1': int(dense1.opt_weights.t),
        'n_biases_layer1': 128,
        'biases_layer1': dense1.biases.tolist(),
        'opt_biases_m_layer1': dense1.opt_biases.m.tolist(),
        'opt_biases_v_layer1': dense1.opt_biases.v.tolist(),
        'opt_biases_t_layer1': int(dense1.opt_biases.t),
        'n_weights_layer2': 128,
        'weights_layer2': dense2.weights.tolist(),
        'opt_weights_m_layer2': dense2.opt_weights.m.tolist(),
        'opt_weights_v_layer2': dense2.opt_weights.v.tolist(),
        'opt_weights_t_layer2': int(dense2.opt_weights.t),
        'n_biases_layer2': 128,
        'biases_layer2': dense2.biases.tolist(),
        'opt_biases_m_layer2': dense2.opt_biases.m.tolist(),
        'opt_biases_v_layer2': dense2.opt_biases.v.tolist(),
        'opt_biases_t_layer2': int(dense2.opt_biases.t),
        'n_weights_layer3': 128,
        'weights_layer3': dense3.weights.tolist(),
        'opt_weights_m_layer3': dense3.opt_weights.m.tolist(),
        'opt_weights_v_layer3': dense3.opt_weights.v.tolist(),
        'opt_weights_t_layer3': int(dense3.opt_weights.t),
        'n_biases_layer3': num_classes,
        'biases_layer3': dense3.biases.tolist(),
        'opt_biases_m_layer3': dense3.opt_biases.m.tolist(),
        'opt_biases_v_layer3': dense3.opt_biases.v.tolist(),
        'opt_biases_t_layer3': int(dense3.opt_biases.t),
    }

    model_params['categories'] = unique_labels

    with open('main/model.json', 'w') as f:
        json.dump(model_params, f)
    end = time.time()
    print(f"Success. This took {(int(end - start))/60} minutes")