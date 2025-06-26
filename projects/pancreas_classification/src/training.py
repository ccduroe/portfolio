import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def train_dnn_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        verbose=1, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])
    return history
