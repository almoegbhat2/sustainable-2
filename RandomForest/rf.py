"""
rf.py

This script trains a Random Forest classifier to predict binary labels ('Overhype') using a preprocessed dataset.
It applies SMOTE for class balancing, feature scaling, and hyperparameter tuning with GridSearchCV.

Dependencies:
    pip install scikit-learn imbalanced-learn
"""

import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def main():
    """
    Loads dataset, applies SMOTE, scales features, trains and tunes a Random Forest classifier.
    """
    with open('repo/Dataset/df_train.pkl', 'rb') as f:
        df_train = pickle.load(f)

    X = df_train.drop(['Overhype'], axis=1)
    Y = df_train['Overhype'].apply(lambda x: x if x in [0, 1] else 0)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X, Y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_resampled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_train_resampled,
        test_size=0.2,
        random_state=42,
        stratify=y_train_resampled
    )

    param_grid = {
        'n_estimators': [300, 500, 1000],
        'max_depth': [24, 28, 32],
        'min_samples_split': [2, 5, 10],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced_subsample", random_state=42),
        param_grid=param_grid,
        cv=10,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    print("Random Forest Trained!")

if __name__ == '__main__':
    main()