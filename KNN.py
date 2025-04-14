import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# ---- Load the dataset ----
df = pd.read_csv("processed_data.csv")

# Separate features and labels
X = df.iloc[:, :-2]  # all columns except last two
y_primary = df["Primary Typing Label"]
y_secondary = df["Secondary Typing Label"]

# ---- Split into train and test BEFORE applying SMOTE ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y_primary, test_size=0.2, random_state=42, stratify=y_primary
)

# ---- Apply SMOTE to training set only ----
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ---- Apply PCA (e.g., reduce to 20 components or less) ----
pca = PCA(n_components=20)  # you can try other values (e.g., 10, 30, etc.)
X_train_pca = pca.fit_transform(X_train_resampled)
X_test_pca = pca.transform(X_test)

# ---- Train KNN model ----
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_pca, y_train_resampled)

# ---- Evaluate on test set ----
y_pred = knn.predict(X_test_pca)
print("=== Classification Report (Primary Type) ===")
print(classification_report(y_test, y_pred))

# ---- Re-run prediction on full dataset for 'either type' check ----
X_full_pca = pca.transform(X)
y_pred_full = knn.predict(X_full_pca)

correct = (y_pred_full == y_primary) | (y_pred_full == y_secondary)
combined_accuracy = correct.sum() / len(correct)
print(f"\nAccuracy (Correct if Prediction Matches Either Type): {combined_accuracy:.4f}")

# ---- Plot Accuracy vs k with PCA + SMOTE ----
print("\n--- Accuracy vs k (matches either type) ---")
ks = []
accuracies = []

for k_try in range(1, 16):
    knn_try = KNeighborsClassifier(n_neighbors=k_try)
    knn_try.fit(X_train_pca, y_train_resampled)
    y_pred_try = knn_try.predict(X_full_pca)
    correct_try = (y_pred_try == y_primary) | (y_pred_try == y_secondary)
    acc = correct_try.sum() / len(correct_try)
    ks.append(k_try)
    accuracies.append(acc)
    print(f"k={k_try:2d} -> Accuracy: {acc:.4f}")

# Optional: Plot the results
plt.plot(ks, accuracies, marker='o')
plt.title("KNN Accuracy vs k (with PCA + SMOTE)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy (Correct if Matches Either Type)")
plt.grid(True)
plt.show()
