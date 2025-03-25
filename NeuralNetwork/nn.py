import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def main():
    with open('./Dataset/df_train.pkl', 'rb') as f:
        df_train = pickle.load(f)

    X = df_train.drop(['Overhype'], axis=1)
    Y = df_train['Overhype'].apply(lambda x: x if x in [0, 1] else 0)

    scaler = StandardScaler()
    smote = SMOTE(random_state=42)

    X_resampled, Y_resampled = smote.fit_resample(X, Y)
    X_scaled = scaler.fit_transform(X_resampled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y_resampled, test_size=0.2, random_state=42, stratify=Y_resampled
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    model = SimpleNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        preds_class = (preds >= 0.5).float().squeeze().numpy()
        y_true = y_test_tensor.squeeze().numpy()

    acc = accuracy_score(y_true, preds_class)
    f1 = f1_score(y_true, preds_class, average='weighted')

    results = {"Accuracy": acc, "F1 Score": f1}
    print(results)


if __name__ == '__main__':
    main()