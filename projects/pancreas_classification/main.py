from src.data_loading import load_data
from src.preprocessing import preprocess_data
from src.models import create_svm_model, create_decision_tree, create_adaboost_model, create_random_forest_model, create_dnn_model
from src.training import train_dnn_model
from src.evaluation import evaluate_model

def main():
    mouse_data_path = 'data/MousePancreas.h5ad'
    human_data_path = 'data/HumanPancreas.h5ad'
    mouse_label_path = 'data/MousePancreaslabel.txt'
    human_label_path = 'data/HumanPancreaslabel.txt'

    mouse_data, human_data, train_label_df, test_label_df = load_data(mouse_data_path, human_data_path, mouse_label_path, human_label_path)

    X_train = mouse_data.X
    X_test = human_data.X
    y_train = train_label_df
    y_test = test_label_df

    X_train_resampled, X_test_scaled, X_train_pca, X_test_pca, y_train_resampled, y_test_encoded, label_encoder = preprocess_data(X_train, X_test, y_train, y_test)

    # Train and evaluate different models
    svm_model = create_svm_model()
    svm_model.fit(X_train_resampled, y_train_resampled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    evaluate_model(y_test_encoded, y_pred_svm, label_encoder)

    dnn_model = create_dnn_model(X_train_resampled.shape[1], len(label_encoder.classes_))
    history = train_dnn_model(dnn_model, X_train_resampled, y_train_resampled, X_test_scaled, y_test_encoded)
    y_pred_dnn = tf.argmax(dnn_model.predict(X_test_scaled), axis=1).numpy()
    evaluate_model(y_test_encoded, y_pred_dnn, label_encoder)

if __name__ == "__main__":
    main()
