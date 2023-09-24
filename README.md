# machine_learning_study

- Linear Regression
    - Binary linear regression
    - Multi-variable linear regression (inputs > 1)
    - Optimizer 
       - Gradient descent method
       - Adams
    - Cost function: Mean squared error
      
- Logistic Regression
    - Binary logistic regression
        - Activation function: Sigmoid function
        - Const function: binary_crossentropy
    - Multinominal Logistic Regression
        - One-hot encoding
        - Activation function: Softmax function
        - Cost function: categorical_crossentropy
  
- Preprocessing
    -  Split dataset into test & validation dataset
        - from sklearn.model_selection import train_test_split
    -  One-hot encoding for multi-labels output
    -  Normalization
    -  Standardization
        - from sklearn.preprocessing import StandardScaler

- Machine Learning Models
    -  Support vector machine (SVM)
    -  k-Nearest neighbors (KNN)
    -  Decision tree
    -  Random forest
        - Majority voting

- Deep Learning
    - MLP
    - Deep feed forward (DFF)
    - Activation function: Relu for hidden layers
    - Activation function: Sigmoid - binary & Softmax - multinomial for output layer
    - Keras Functional API
    - MNIST Dataset

- Overfitting/ underfitting
    - Data augmentation
    - Dropout
    - Ensemble
    - Learning rate decay (Learning rate schedules)
        - tf.keras.callbacks.LearningRateScheduler() & tf.keras.callbacks.ReduceLROnPlateau()

-  Neural Network
      - Convolutional Neural Networks (CNN)
        - Convolution layer, Pooling layer, Flatten layer, Dense (fully connected) layer
      - Recurrent Neural Networks (RNN)
      - Generative Adversarial Network (GAN)
- Data Generator
- Image augmentation & preprocess
- Transfer Learning
      - Inception V3
          -  Inception module
      - ResNet
          - Residual block
              - Shortcut=Skip connection)
              - To solve gradient vanishing

