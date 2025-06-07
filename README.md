# Deep Learning with Keras

## An introduction to Deep Learning &amp; Neural Networks with Keras

This repository provides a practical introduction to the foundational concepts of deep learning, using Keras to illustrate how to build and train neural networks effectively. It covers essential topics such as gradient descent, backpropagation, activation functions, and data preparation for classification tasks.

---

### Gradient Descent and the Training Loop

**Gradient Descent** is an iterative optimization algorithm used to minimize the model’s error. A key hyperparameter in this process is the **learning rate**:
- If the learning rate is **too high**, the model may overshoot the minimum and fail to converge.
- If it is **too low**, training becomes slow and may get stuck in suboptimal points.

Training a neural network involves repeatedly cycling through three main steps:
1. **Forward propagation** – where the input is passed through the network to generate a prediction.
2. **Error calculation** – comparing the predicted output with the true value.
3. **Backpropagation** – updating weights and biases to reduce the error.

This loop continues over multiple iterations (called **epochs**) until a stopping criterion is met, such as a low enough error or a maximum number of epochs.

📘 Notebook:  [Backpropagation](https://github.com/ElsonFilho/DeepLearning_Keras/blob/main/notebooks/BackProp.ipynb)  

---

### ⚠️ Vanishing Gradient Problem

A significant challenge in training deep networks is the **vanishing gradient problem**, especially when using activation functions like **sigmoid**. In deep architectures:
- Gradients in early layers can become extremely small during backpropagation.
- This results in **slow or stalled learning** in those layers.
- Ultimately, this affects the model's ability to learn complex representations and degrades accuracy.

To mitigate this, activation functions that avoid shrinking gradients — such as **ReLU** — are preferred in modern architectures, especially in hidden layers.

---

### Activation Functions in Neural Networks

Activation functions introduce non-linearity into the network, enabling it to model complex relationships. Common types include:
- **Sigmoid**: Historically popular but now less used due to vanishing gradients.
- **Tanh**: A centered and scaled version of sigmoid, also prone to gradient issues.
- **ReLU (Rectified Linear Unit)**: The most widely used function today; it enables faster, more efficient training by avoiding activation of all neurons at once.
- **Softmax**: Used in the output layer for multi-class classification tasks, converting outputs into probability distributions.

📘 Notebook: [Activation Functions](https://github.com/ElsonFilho/DeepLearning_Keras/blob/main/notebooks/Activation_Functions.ipynb)

---

### Deep Learning Libraries: TensorFlow, PyTorch & Keras

There are several major libraries used for deep learning development:

- **TensorFlow**: Ideal for deploying models in production; backed by a strong community.
- **PyTorch**: Preferred in academic research; known for flexibility and strong GPU support.
- Both are powerful but can be challenging for beginners due to their complexity.

- **Keras**: A high-level API that simplifies the process of building and training deep learning models. It is especially beginner-friendly, offering:
  - Clean and readable syntax,
  - Rapid prototyping,
  - Integration with TensorFlow as its backend.

Keras enables the creation of powerful models with just a few lines of code, making it an excellent choice for those new to deep learning.

---

### Data Preparation for Keras Models

Before training a model with Keras, your dataset must be properly structured:
- Split your data into **predictors (input features)** and **target (labels)**.
- For **classification tasks**, the target values must be transformed into binary arrays using the `to_categorical()` function from Keras utilities.

Once prepared, the data can be fed into Keras models to build and train deep learning architectures efficiently.

📘 Notebook: [Intro to Keras](https://github.com/ElsonFilho/DeepLearning_Keras/blob/main/notebooks/Keras_Intro.ipynb)

---

## Deep Learning Models

### Shallow vs. Deep Neural Networks
- **Shallow Neural Networks**: One hidden layer; only accept vector inputs.
- **Deep Neural Networks**: Multiple hidden layers; can process raw data like images and text.
- **Rise of Deep Learning** driven by: algorithmic advancements, large data availability, and powerful hardware (e.g., GPUs).
  
---

### Convolutional Neural Networks (CNNs)
- Designed specifically for image-related tasks (e.g., recognition, object detection).
- Input shapes: 
  - Grayscale: `(n x m x 1)`
  - RGB/Color: `(n x m x 3)`
- Layers:
  - **Convolutional Layer**: Applies filters (kernels) to extract features.
  - **ReLU Activation**: Keeps positive values, zeroes out negatives.
  - **Pooling Layer**: Reduces spatial dimensions (Max Pooling, Avg Pooling).
  - **Fully Connected Layer**: Flattens feature maps and connects to output.

📘 Notebook: [CNNs with Keras](https://github.com/ElsonFilho/DeepLearning_Keras/blob/main/notebooks/CNN_Keras.ipynb)

---


### Recurrent Neural Networks (RNNs)
- Handle **sequential data** by incorporating past outputs into current inputs.
- Suitable for tasks like:
  - Text & speech processing
  - Time series prediction
  - Handwriting and genome analysis
- **LSTM (Long Short-Term Memory)**: A type of RNN that captures long-term dependencies. Used for:
  - Image captioning
  - Handwriting/image generation
  - Video description
  
---

### Transformers
- Handle **sequential data** using **self-attention**, not recurrence.
- Capture relationships between all tokens in a sequence simultaneously.
- Enable highly parallelizable and scalable training.
- Foundation of modern NLP models like **BERT**, **GPT**, and **T5**.
- Used in tasks such as language translation, summarization, and question answering.

📘 Notebook: [Transformers with Keras](https://github.com/ElsonFilho/DeepLearning_Keras/blob/main/notebooks/Transformers_Keras.ipynb)

---

### Autoencoders & RBMs
- **Autoencoders**: Unsupervised models for data compression and reconstruction.
  - Encoder compresses input; decoder reconstructs it.
  - Applications: noise removal, dimensionality reduction, data visualization.
- **Restricted Boltzmann Machines (RBMs)**:
  - Used for feature extraction, handling imbalanced data, estimating missing values.
