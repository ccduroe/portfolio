from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model(y_test, y_pred, label_encoder):
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
