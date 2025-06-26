from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import numpy as np

def preprocess_data(X_train, X_test, y_train, y_test):
    # Filter common labels
    train_unique_labels = set(y_train["Label"])
    test_unique_labels = set(y_test["Label"])
    common_labels = train_unique_labels.intersection(test_unique_labels)

    train_common_indices = y_train[y_train["Label"].isin(common_labels)].index
    test_common_indices = y_test[y_test["Label"].isin(common_labels)].index

    X_train_filtered = X_train[train_common_indices]
    X_test_filtered = X_test[test_common_indices]
    y_train_filtered = y_train.loc[train_common_indices]
    y_test_filtered = y_test.loc[test_common_indices]

    # Convert labels to numerical format
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_filtered["Label"])
    y_test_encoded = label_encoder.transform(y_test_filtered["Label"])

    # Filter common features
    train_feature_names = np.unique(X_train_filtered.flatten())
    test_feature_names = np.unique(X_test_filtered.flatten())
    common_feature_names = np.intersect1d(train_feature_names, test_feature_names)

    train_common_indices = [i for i, feature in enumerate(X_train[0]) if feature in common_feature_names]
    X_train_filtered_features = X_train_filtered[:, train_common_indices]
    X_test_filtered_features = X_test_filtered[:, train_common_indices]

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_filtered_features)
    X_test_scaled = scaler.transform(X_test_filtered_features)

    # Balance data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_encoded)

    # Apply PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_resampled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_resampled, X_test_scaled, X_train_pca, X_test_pca, y_train_resampled, y_test_encoded, label_encoder
