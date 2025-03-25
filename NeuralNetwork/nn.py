import pickle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras import layers
import numpy as np


def main():
    with open('repo/Dataset/df_train.pkl', 'rb') as f:
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

    input_dim = X_train.shape[1]
    tf.random.set_seed(42)
    np.random.seed(42)

    print("Building model...")
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

if __name__ == '__main__':
    main()
