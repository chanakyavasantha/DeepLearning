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
