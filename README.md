# Machine Learning Projects

This repository contains three machine learning projects implemented in Python. Each project focuses on different aspects of machine learning and natural language processing.

## Project 1: Decision Tree Classifier for Iris Dataset

### Overview
This project implements a Decision Tree Classifier to classify iris flowers using the famous Iris dataset. The implementation uses scikit-learn and includes comprehensive data visualization and model evaluation.

### Features
- Data preprocessing and exploration
- Decision Tree model implementation
- Model evaluation with multiple metrics
- Visualization of results using seaborn and matplotlib

### Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Implementation Details
1. **Data Loading and Preprocessing**
   - Loads the Iris dataset using sklearn.datasets
   - Converts data to pandas DataFrame
   - Handles missing values and data normalization

2. **Data Exploration**
   - Displays dataset statistics
   - Creates pairplots and correlation heatmaps
   - Shows class distribution

3. **Model Training**
   - Splits data into training and testing sets
   - Implements Decision Tree Classifier
   - Tunes hyperparameters for optimal performance

4. **Model Evaluation**
   - Calculates accuracy, precision, and recall
   - Generates classification report
   - Creates confusion matrix visualization

## Project 2: MNIST Digit Classification using PyTorch

### Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset.

### Features
- CNN architecture implementation
- Data preprocessing and augmentation
- Model training and evaluation
- Visualization of predictions

### Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- seaborn
- scikit-learn

### Implementation Details
1. **Data Preparation**
   - Loads MNIST dataset using torchvision
   - Normalizes pixel values
   - Creates DataLoader for batch processing

2. **Model Architecture**
   - Implements CNN with multiple convolutional layers
   - Uses ReLU activation and MaxPooling
   - Includes dropout for regularization
   - Final layer uses softmax for classification

3. **Training Process**
   - Implements training loop with batch processing
   - Uses CrossEntropyLoss and Adam optimizer
   - Tracks training metrics
   - Implements early stopping

4. **Evaluation and Visualization**
   - Calculates test accuracy
   - Generates confusion matrix
   - Visualizes sample predictions
   - Plots training history

### Alternative Implementation: TensorFlow/Keras

#### Overview
This project also provides an alternative implementation using TensorFlow/Keras for MNIST digit classification.

#### Features
- CNN architecture using Keras Sequential API
- Built-in data preprocessing
- Model training with callbacks
- Comprehensive evaluation metrics

#### Requirements
- Python 3.x
- TensorFlow
- Keras
- numpy
- matplotlib
- seaborn
- scikit-learn

#### Implementation Details
1. **Data Preparation**
   - Loads MNIST dataset using keras.datasets
   - Normalizes pixel values (0-1)
   - Reshapes data for CNN input
   - One-hot encodes labels

2. **Model Architecture**
   - Uses Keras Sequential API
   - Multiple Conv2D layers with ReLU activation
   - MaxPooling2D layers for dimensionality reduction
   - Dense layers with dropout for classification
   - Softmax activation in output layer

3. **Training Process**
   - Uses categorical crossentropy loss
   - Adam optimizer with learning rate scheduling
   - Implements callbacks for:
     - Early stopping
     - Model checkpointing
     - Learning rate reduction
   - Batch processing with validation split

4. **Evaluation and Visualization**
   - Calculates test accuracy and loss
   - Generates classification report
   - Creates confusion matrix
   - Visualizes training history
   - Shows sample predictions with true/predicted labels

## Project 3: Named Entity Recognition and Sentiment Analysis

### Overview
This project performs Named Entity Recognition (NER) and sentiment analysis on Amazon product reviews using spaCy and VADER sentiment analyzer.

### Features
- Entity extraction from product reviews
- Sentiment analysis using VADER
- Visualization of results
- Comprehensive analysis of extracted entities

### Requirements
- Python 3.x
- spaCy
- vaderSentiment
- pandas
- matplotlib
- seaborn

### Implementation Details
1. **Data Preparation**
   - Sample Amazon product reviews
   - Text preprocessing
   - Entity extraction setup

2. **Entity Recognition**
   - Uses spaCy's NER capabilities
   - Extracts product and brand entities
   - Categorizes entities by type

3. **Sentiment Analysis**
   - Implements VADER sentiment analyzer
   - Calculates compound sentiment scores
   - Classifies sentiment as positive, negative, or neutral

4. **Results Analysis**
   - Creates visualizations of sentiment distribution
   - Generates summary statistics
   - Displays extracted entities
   - Provides comprehensive analysis of results

## Usage

Each project can be run independently. Follow these steps to run any project:

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. For Project 2 (PyTorch), install PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. For Project 3, install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Run the desired project's notebook in Jupyter or Google Colab.

## Notes
- Each project includes detailed comments and documentation
- Code is modular and reusable
- Visualizations are included for better understanding
- Results are clearly presented with appropriate metrics
