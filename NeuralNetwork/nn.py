import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def main():
    with open('../Dataset/df_train.pkl', 'rb') as f:
        df_train = pickle.load(f)

    X = df_train.drop(['Overhype'], axis=1)
    Y = df_train['Overhype']
    Y = Y.apply(lambda x: x if x in [0, 1] else 0)
    scaler = StandardScaler()

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X, Y)
    X_scaled = scaler.fit_transform(X_train_resampled)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_train_resampled, test_size=0.2, random_state=42,
                                                        stratify=y_train_resampled)

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=5,
                                class_weight="balanced_subsample", random_state=42)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    results = {"Accuracy": accuracy, "F1 Score": f1}
    print(results)


if __name__ == '__main__':
    main()
