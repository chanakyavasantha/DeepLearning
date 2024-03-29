# Learn Deep Learning
### DL learning resources and projects
<hr>

#### **Machine Learning** :
Development of Algorithmns and computer systems that can learn patterns from the data and are able to predict outcomes without explicit instructions.
#### How is ML different from tradiitional programming:
1. **Approach to Problem Solving:**
   - **Traditional Programming:** Developers manually write code with step-by-step instructions.
   - **Machine Learning:** Algorithms learn from data to make predictions without explicit programming.

2. **Data-Driven vs. Rule-Driven:**
   - **Traditional Programming:** Relies on predefined rules and logic.
   - **Machine Learning:** Learns patterns from data for making predictions.

3. **Adaptability:**
   - **Traditional Programming:** Requires manual code updates for changes.
   - **Machine Learning:** Adapts to new data and situations without code changes.

4. **Complexity and Scaling:**
   - **Traditional Programming:** Can become complex and challenging for large-scale problems.
   - **Machine Learning:** Handles complexity and scales well for large datasets.

5. **Domain Expertise:**
   - **Traditional Programming:** Requires deep domain knowledge to define accurate rules.
   - **Machine Learning:** Learns from domain-specific data, capturing complex patterns.

6. **Problem Types:**
   - **Traditional Programming:** Suited for tasks with clear rules.
   - **Machine Learning:** Suited for tasks with complex patterns, noise, or ambiguity.

7. **Feedback Loop:**
   - **Traditional Programming:** Requires manual code updates based on feedback.
   - **Machine Learning:** Improves over time with more data and model updates.

In summary, traditional programming uses explicit rules, while machine learning learns from data for making predictions. ML excels in complex, data-driven tasks, but requires careful data handling and model tuning.
<hr>

**Deep Learning** is a subset of machine learning that involves training artificial neural networks to perform tasks by learning from large amounts of data. It aims to simulate the human brain's structure and function in order to enable machines to learn and make decisions on their own. Deep learning algorithms consist of multiple layers of interconnected nodes (neurons) that process and transform data.

Key characteristics of deep learning:

1. **Multiple Layers:** Deep learning models typically consist of many layers, including an input layer, multiple hidden layers, and an output layer. These layers allow the model to learn increasingly abstract features from the data as information flows through the network.

2. **Feature Learning:** Deep learning models automatically learn relevant features or representations from the data. This eliminates the need for manual feature engineering, as the model learns to extract important patterns directly from raw input.

3. **Neural Networks:** Deep learning models are based on artificial neural networks, which are inspired by the structure and functioning of biological neurons. These networks are designed to process and transform data in a way that captures complex relationships and patterns.

4. **Hierarchical Representation:** Information is processed hierarchically in deep learning models, with lower layers capturing simple features (edges, textures) and higher layers capturing more complex and abstract features (shapes, objects).

5. **Training with Backpropagation:** Deep learning models are trained using a technique called backpropagation. During training, the model's predictions are compared to the actual outcomes, and the model adjusts its internal parameters (weights and biases) to minimize the prediction error.

6. **Large Datasets:** Deep learning requires large amounts of labeled data for training. The availability of big data has contributed to the success of deep learning algorithms.

7. **Variety of Applications:** Deep learning has achieved remarkable success in various fields, including image and speech recognition, natural language processing, autonomous vehicles, medical image analysis, and more.

8. **Complexity and Computational Resources:** Deep learning models can be highly complex and may require significant computational resources for training, often utilizing specialized hardware like Graphics Processing Units (GPUs) or specialized hardware accelerators.

Popular types of deep learning architectures include Convolutional Neural Networks (CNNs) for image analysis, Recurrent Neural Networks (RNNs) for sequence data, and Transformers for natural language processing.

It's important to note that while deep learning has achieved impressive results in various domains, it also requires careful tuning, training, and validation to ensure optimal performance and avoid overfitting (when the model performs well on the training data but poorly on new, unseen data).

<hr>

#### What is an ANN:
An **Artificial Neural Network (ANN)** is a computational model inspired by the brain's structure. It consists of layers of interconnected nodes (neurons) and is used in machine learning for tasks like pattern recognition and decision-making. Neurons process inputs using weights and activation functions. Training adjusts weights to optimize performance. ANN has led to breakthroughs in various fields, like image recognition and language processing.

#### Implement your first neural network using tensorflow's sequential API:
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Add a Dense layer with 64 units and ReLU activation for the input layer
model.add(Dense(units=64, activation='relu', input_dim=8))  # Adjust input_dim to match your data

# Add a second Dense layer with 32 units and ReLU activation
model.add(Dense(units=32, activation='relu'))

# Add the output layer with 1 unit and sigmoid activation for binary classification
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()
```

In an Artificial Neural Network (ANN), various types of layers can be used to process and transform data. Each type of layer serves a specific purpose in capturing different types of patterns and relationships in the data. Here are some common types of layers found in ANNs:

1. **Dense (Fully Connected) Layer:**
   - The dense layer is the basic building block of an ANN. Each neuron in this layer is connected to every neuron in the previous layer, and each connection has an associated weight.
   - Dense layers are often used for capturing complex patterns and relationships in the data.

2. **Convolutional Layer (Convolutional Neural Networks - CNNs):**
   - Convolutional layers are primarily used for processing grid-like data, such as images. They apply filters (kernels) to detect local patterns and features in the input data.
   - CNNs are highly effective for tasks like image recognition and computer vision.

3. **Pooling Layer (CNNs):**
   - Pooling layers reduce the spatial dimensions of the data by down-sampling, helping to decrease computation and control overfitting.
   - Common pooling operations include max pooling and average pooling.

4. **Recurrent Layer (Recurrent Neural Networks - RNNs):**
   - Recurrent layers, such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) layers, are designed to handle sequential data by maintaining an internal state that captures temporal dependencies.
   - RNNs are well-suited for tasks involving sequences, such as natural language processing and time series prediction.

5. **Embedding Layer:**
   - Embedding layers map categorical variables (like words or categories) to continuous vector representations, allowing the network to learn meaningful relationships between them.
   - Often used in natural language processing for word embeddings.

6. **Normalization Layer (Batch Normalization):**
   - Normalization layers standardize the inputs to a layer, which can accelerate training and improve convergence.
   - Batch normalization is a common form of normalization used to stabilize training.

7. **Dropout Layer:**
   - Dropout layers randomly deactivate a fraction of neurons during training, helping to prevent overfitting by improving the network's generalization ability.

8. **Flatten Layer:**
   - The flatten layer reshapes higher-dimensional data into a 1D vector, often used to transition from convolutional layers to fully connected layers.

9. **Output Layer:**
   - The output layer produces the final prediction or output of the network, depending on the task. The architecture of this layer varies, such as using sigmoid activation for binary classification or softmax for multi-class classification.

10. **Other Specialized Layers:**
   - Depending on the problem, there are various specialized layers, such as attention layers used in transformer models for natural language processing, and more.

These are just a few examples of the types of layers that can be used in ANNs. The choice of layers depends on the problem at hand and the characteristics of the data. ANN architectures often combine different layer types to effectively capture and process various patterns in the data.

<hr>

#### Activation Functions in ANN:
![Screenshot from 2023-08-12 14-23-25](https://github.com/chanakyavasantha/DeepLearning/assets/93817654/1fce38e7-0809-4a6f-b02e-f16f350ceed7)

![Screenshot from 2023-08-12 14-29-44](https://github.com/chanakyavasantha/DeepLearning/assets/93817654/95f56d9d-0dc7-4f11-bf4a-758950ed7314)

Activation functions are a crucial component in artificial neural networks and deep learning models. They introduce non-linearity to the network, enabling it to learn and approximate complex functions. Different activation functions have specific properties that make them suitable for various use cases. Here are some common activation functions and their use cases:

1. **Sigmoid Activation (Logistic Function):**
   - Use Case: Historically used in binary classification problems, where the output needs to be between 0 and 1. Also used in the hidden layers of shallow networks.
   - Note: Can suffer from vanishing gradient problem, limiting its effectiveness in deep networks.

2. **Hyperbolic Tangent (Tanh) Activation:**
   - Use Case: Similar to sigmoid but with a range between -1 and 1. Suitable for situations where data is centered around zero and requires normalization.
   - Note: Still susceptible to vanishing gradient issues.

3. **Rectified Linear Unit (ReLU) Activation:**
   - Use Case: Extremely popular due to its simplicity and effectiveness. Widely used in deep neural networks for hidden layers. Helps mitigate vanishing gradient issues.
   - Note: Can suffer from "dying ReLU" problem where some units become inactive and do not update during training.

4. **Leaky ReLU Activation:**
   - Use Case: Variation of ReLU to address the "dying ReLU" problem. Allows a small gradient when the unit is not active, promoting better training.
   - Note: Prevents some of the issues with ReLU but may still suffer from vanishing gradient for very negative inputs.

5. **Parametric ReLU (PReLU) Activation:**
   - Use Case: Extension of Leaky ReLU where the slope of the negative side is learned during training. Can help with convergence and performance.
   - Note: Adds an additional parameter to the network, which might increase complexity.

6. **Exponential Linear Unit (ELU) Activation:**
   - Use Case: Addresses the limitations of ReLU and its variations by reducing the "dying ReLU" problem and enabling negative values without gradient issues.
   - Note: May require additional computational resources compared to simpler activations.

7. **Scaled Exponential Linear Unit (SELU) Activation:**
   - Use Case: Designed for self-normalizing networks, it maintains mean and variance of activations, potentially leading to faster convergence and improved performance.
   - Note: Requires specific weight initialization and may not work optimally in all scenarios.

8. **Softmax Activation:**
   - Use Case: Typically used in the output layer of multi-class classification problems. Converts raw scores into probability distributions, aiding in class selection.
   - Note: Assumes mutually exclusive classes and is not suitable for regression tasks.

9. **Linear Activation (Identity Function):**
   - Use Case: Used in regression problems where the output needs to be directly proportional to the input. Also used in the output layer of neural networks for regression tasks.

10. **Swish Activation:**
    - Use Case: A newer activation function that combines the benefits of ReLU and Sigmoid, potentially offering better training and generalization.

The choice of activation function depends on factors such as the problem type, network architecture, potential vanishing/exploding gradient issues, and empirical performance on the specific task. Experimentation and fine-tuning are often required to find the most suitable activation for a given model.


