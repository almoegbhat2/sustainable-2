"""
nn.py

This script trains a deep neural network (SimpleNN) to predict binary labels ('Overhype') using a preprocessed dataset.
It includes SMOTE for handling class imbalance and uses PyTorch for modeling.

Dependencies:
    pip install torch scikit-learn imbalanced-learn
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

class SimpleNN(nn.Module):
    """
    A fully-connected deep neural network for binary classification.
    Architecture: 10 linear layers with ReLU and dropout.
    """
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def main():
    """
    Loads the dataset, applies preprocessing, trains the neural network model, and prints training loss.
    """
    # Load dataset
    with open('repo/Dataset/df_train.pkl', 'rb') as f:
        df_train = pickle.load(f)

    # Extract features and target
    X = df_train.drop(['Overhype'], axis=1)
    Y = df_train['Overhype'].apply(lambda x: x if x in [0, 1] else 0)

    # Resample and scale
    scaler = StandardScaler()
    smote = SMOTE(random_state=42)
    X_resampled, Y_resampled = smote.fit_resample(X, Y)
    X_scaled = scaler.fit_transform(X_resampled)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y_resampled, test_size=0.2, random_state=42, stratify=Y_resampled
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Model setup
    model = SimpleNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()