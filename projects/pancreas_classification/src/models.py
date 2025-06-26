from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

def create_svm_model(kernel='linear'):
    return SVC(kernel=kernel, probability=True, random_state=42)

def create_decision_tree(max_depth=None):
    return DecisionTreeClassifier(max_depth=max_depth, random_state=42)

def create_adaboost_model():
    return AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5)

def create_random_forest_model():
    return RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)

def create_dnn_model(input_shape, num_classes):
    model = Sequential([
        Dense(500, LeakyReLU(alpha=0.1), input_shape=(input_shape,)),
        Dense(250, LeakyReLU(alpha=0.1)),
        Dense(125, LeakyReLU(alpha=0.1)),
        Dense(75, LeakyReLU(alpha=0.1)),
        Dense(50, LeakyReLU(alpha=0.1)),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
