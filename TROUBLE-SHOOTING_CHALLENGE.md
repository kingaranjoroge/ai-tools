# MNIST CNN: Bug Fixes and Corrections

This document shows a buggy TensorFlow implementation of a CNN for MNIST digit classification, followed by the corrected version with explanations of each fix.

## Buggy Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Bug 1: Missing data normalization
# Bug 2: Missing reshape for CNN input
x_train = x_train
x_test = x_test

# Bug 3: Incorrect model architecture
model = models.Sequential([
    # Bug 4: Missing input shape specification
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Bug 5: Missing Flatten layer
    layers.Dense(64, activation='relu'),
    # Bug 6: Wrong output layer activation and units
    layers.Dense(1, activation='sigmoid')
])

# Bug 7: Wrong loss function for multi-class classification
# Bug 8: Missing metrics
model.compile(optimizer='adam',
              loss='binary_crossentropy')

# Bug 9: Missing validation split
# Bug 10: Missing batch size
model.fit(x_train, y_train, epochs=5)

# Bug 11: Missing model evaluation
```

## Corrected Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Fix 1: Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Fix 2: Reshape data for CNN input (samples, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Fix 3: Correct model architecture
model = models.Sequential([
    # Fix 4: Specify input shape
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    # Fix 5: Add Flatten layer before Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Added dropout for regularization
    # Fix 6: Correct output layer (10 units for digits 0-9, softmax activation)
    layers.Dense(10, activation='softmax')
])

# Fix 7: Use correct loss function for multi-class classification
# Fix 8: Add appropriate metrics
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fix 9: Add validation split
# Fix 10: Specify batch size
history = model.fit(x_train, y_train,
                   batch_size=64,
                   epochs=10,
                   validation_split=0.2,
                   verbose=1)

# Fix 11: Evaluate model on test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
```

## Explanation of Fixes

1. **Data Normalization**
   - Bug: Missing normalization of pixel values
   - Fix: Divide by 255.0 to scale values to [0, 1]
   - Why: Neural networks train better with normalized input data

2. **Input Shape**
   - Bug: Missing reshape for CNN input
   - Fix: Reshape to (samples, height, width, channels)
   - Why: CNNs expect 4D input tensors

3. **Model Architecture**
   - Bug: Incorrect layer structure
   - Fix: Added proper CNN architecture with multiple Conv2D layers
   - Why: Deeper networks can learn more complex features

4. **Input Shape Specification**
   - Bug: Missing input shape in first Conv2D layer
   - Fix: Added input_shape=(28, 28, 1)
   - Why: First layer needs to know input dimensions

5. **Flatten Layer**
   - Bug: Missing Flatten layer
   - Fix: Added Flatten() before Dense layers
   - Why: Converts 2D feature maps to 1D for Dense layers

6. **Output Layer**
   - Bug: Wrong units and activation
   - Fix: Changed to 10 units with softmax activation
   - Why: 10 classes (digits 0-9) need softmax for multi-class probabilities

7. **Loss Function**
   - Bug: Using binary_crossentropy
   - Fix: Changed to sparse_categorical_crossentropy
   - Why: Appropriate for multi-class classification with integer labels

8. **Metrics**
   - Bug: Missing metrics
   - Fix: Added accuracy metric
   - Why: Helps monitor training progress

9. **Validation Split**
   - Bug: Missing validation data
   - Fix: Added validation_split=0.2
   - Why: Helps detect overfitting

10. **Batch Size**
    - Bug: Missing batch size
    - Fix: Added batch_size=64
    - Why: Controls memory usage and training stability

11. **Model Evaluation**
    - Bug: Missing evaluation
    - Fix: Added model.evaluate()
    - Why: Measures model performance on test data

## Additional Improvements

1. **Added Dropout**
   - Purpose: Prevents overfitting
   - Implementation: Added Dropout(0.5) layer

2. **Increased Epochs**
   - Purpose: Allow model to train longer
   - Implementation: Changed epochs from 5 to 10

3. **Added Verbose Parameter**
   - Purpose: Better training progress monitoring
   - Implementation: Added verbose=1 to fit()

These corrections should result in a model that achieves >95% test accuracy on the MNIST dataset. 