import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

##############################################################################
# 1. LOAD & PREPARE DATA
##############################################################################

# Load dataset
df = pd.read_csv("processed_data.csv")

# Replace missing secondary types with -1 (indicates mono-type Pokémon)
df['Secondary Typing Label'] = df['Secondary Typing Label'].fillna(-1)

# Extract features and labels
X = df.drop(['Primary Typing Label', 'Secondary Typing Label'], axis=1)
y_primary = df['Primary Typing Label'].values
y_secondary = df['Secondary Typing Label'].values

# Train-test split (stratify by primary label)
X_train, X_test, y_train, y_test, sec_train, sec_test = train_test_split(
    X, y_primary, y_secondary,
    test_size=0.3,
    random_state=42,
    stratify=y_primary
)

##############################################################################
# 2. BUILD PIPELINE (Scaler → PCA → KNN)
##############################################################################

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', KNeighborsClassifier())
])

# Parameter grid for PCA and KNN
param_grid = {
    'pca__n_components': [0.85, 0.9, 0.95],
    'classifier__n_neighbors': [1, 3, 5, 7],
    'classifier__weights': ['uniform', 'distance'],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
}

# Grid search using primary label accuracy only
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("Starting grid search...")
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV Score (primary label): {grid_search.best_score_:.4f}")

# Get the best pipeline
best_model = grid_search.best_estimator_

##############################################################################
# 3. EVALUATE ON TEST SET (PRIMARY OR SECONDARY)
##############################################################################

# Predict using the best model
y_pred = best_model.predict(X_test)

# Evaluate: match with primary only
acc_primary = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy (Primary Only): {acc_primary:.4f}")

# Evaluate: match with primary OR secondary
correct_either = sum(
    (y_pred[i] == y_test[i]) or (y_pred[i] == sec_test[i])
    for i in range(len(y_pred))
)
acc_either = correct_either / len(y_pred)
print(f"Test Accuracy (Primary OR Secondary): {acc_either:.4f}")

##############################################################################
# 4. PER-TYPE ANALYSIS (OPTIONAL)
##############################################################################

results_df = pd.DataFrame({
    'predicted': y_pred,
    'primary': y_test,
    'secondary': sec_test
})

print("\nPer-primary-type accuracy (primary OR secondary match):")
for ptype, group in results_df.groupby('primary'):
    total = len(group)
    correct = sum(
        (row['predicted'] == row['primary']) or (row['predicted'] == row['secondary'])
        for _, row in group.iterrows()
    )
    print(f"  Type {ptype}: {correct}/{total} correct → {correct/total:.4f}")

##############################################################################
# 5. PCA COMPONENT ANALYSIS (OPTIONAL)
##############################################################################

pca_model = best_model.named_steps['pca']
if hasattr(pca_model, 'components_'):
    print(f"\nNumber of PCA Components Used: {pca_model.n_components_}")

    feature_names = X.columns
    components_df = pd.DataFrame(
        np.abs(pca_model.components_),
        columns=feature_names
    )

    for i in range(min(5, pca_model.n_components_)):
        top_features = components_df.iloc[i].sort_values(ascending=False).head(3)
        print(f"PC{i+1} Top Features: " +
              ", ".join([f"{feat} ({val:.3f})" for feat, val in top_features.items()]))
