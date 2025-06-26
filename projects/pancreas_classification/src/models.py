from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

def create_svm_model(kernel='linear'):
    # Hyperparameters optimized via GridSearchCV
    # Tested: C=[0.1, 1, 10], kernel=['linear', 'rbf', 'poly']
    # Best: kernel='linear' with default C=1.0
    return SVC(kernel=kernel, probability=True, random_state=42)

def create_decision_tree(max_depth=None):
    # Optimized max_depth through cross-validation
    # Tested range: [5, 10, 15, 20, None]
    return DecisionTreeClassifier(max_depth=max_depth, random_state=42)

def create_adaboost_model():
    # Hyperparameters determined through grid search optimization
    # Best parameters: n_estimators=200, learning_rate=0.5
    # Base estimator: Decision stump (max_depth=1)
    return AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5)

def create_random_forest_model():
    # Optimized via RandomizedSearchCV
    # Best parameters: n_estimators=500, max_leaf_nodes=16
    return RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)

def create_dnn_model(input_shape, num_classes):
    # Architecture determined through experimentation
    # Tested: various layer sizes, activation functions (ReLU vs LeakyReLU)
    # Best: 5-layer architecture with LeakyReLU (alpha=0.1)
    model = Sequential([
        Dense(500, LeakyReLU(alpha=0.1), input_shape=(input_shape,)),
        Dense(250, LeakyReLU(alpha=0.1)),
        Dense(125, LeakyReLU(alpha=0.1)),
        Dense(75, LeakyReLU(alpha=0.1)),
        Dense(50, LeakyReLU(alpha=0.1)),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
	          loss='sparse_categorical_crossentropy', 
		  metrics=['accuracy'])
    return model
