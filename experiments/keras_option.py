import numpy as np
import keras
from keras.datasets import mnist

# ── Data ──────────────────────────────────────────────────────────────────────
#
# Load and preprocess MNIST dataset, matching NeuralNetwork.py lines 241-256

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Keras handles one-hot encoding internally with sparse_categorical_crossentropy,
# but we'll use categorical_crossentropy to match our from-scratch implementation
Y_train = keras.utils.to_categorical(y_train, 10)
Y_test = keras.utils.to_categorical(y_test, 10)

# ── Model ─────────────────────────────────────────────────────────────────────
#
# Matches NeuralNetwork.py line 271: [784, 128, 'dropout', 64, 'dropout', 10]
#
# keras.Sequential  →  our Network class
# keras.layers.Dense(units)  →  our Layer class
# keras.layers.LeakyReLU(0.01)  →  our relu() with alpha=0.01
# keras.layers.Dropout(0.2)  →  our DropoutLayer with rate=0.2
#
# He initialization is Keras's default for layers with ReLU-like activations,
# matching our np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)

model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(784,), kernel_initializer='he_normal'),
    keras.layers.LeakyReLU(0.01),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, kernel_initializer='he_normal'),
    keras.layers.LeakyReLU(0.01),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, kernel_initializer='he_normal'),
    keras.layers.Softmax(),
])

# ── Loss & optimizer ──────────────────────────────────────────────────────────
#
# model.compile() sets loss + optimizer + metrics
#
# Adam(learning_rate=0.001) matches our Adam optimizer from line 272
# categorical_crossentropy matches our categorical_cross_entropy from line 232
# accuracy metric matches our accuracy function from line 264

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Training ──────────────────────────────────────────────────────────────────
#
# model.fit() replaces our entire training loop (lines 276-292)
# It handles: forward pass, loss, backward pass, weight updates, batching
#
# epochs=20 matches line 273
# batch_size=64 matches line 274
# validation_data enables test accuracy tracking during training

history = model.fit(
    X_train, Y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_test, Y_test),
    verbose=0
)

# Print results matching our output format from line 293
for epoch in range(25):
    train_loss = history.history['loss'][epoch]
    test_acc = history.history['val_accuracy'][epoch]
    print(f"Epoch {epoch+1:2d} — Loss: {train_loss:.4f}  Test accuracy: {test_acc*100:.1f}%")

# ── Results ───────────────────────────────────────────────────────────────────
#
# model.evaluate() computes final test metrics
# Keras automatically uses eval mode (dropout disabled)

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")