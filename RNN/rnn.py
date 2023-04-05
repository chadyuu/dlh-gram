import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming `X` contains the tokenized and padded input sequences and `y` contains the labels
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a custom dataset for the heart failure data
class HeartFailureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HeartFailureDataset(X_train, y_train)
val_dataset = HeartFailureDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the RNN_GRU model (from the previous answer)
class RNN_GRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(RNN_GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.gru(x)
        return output, hidden

# Hyperparameters
vocab_size = len(your_vocabulary)  # Replace with the size of your vocabulary HW3 uses 128.
embedding_size = k                 # Replace with the desired embedding size. HW3 uses 128.
hidden_size = 128                  # Replace with the desired hidden size

# Initialize the model, loss function, and optimizer
model = RNN_GRU(vocab_size, embedding_size, hidden_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs[:, -1], labels.float())
        loss.backward()
        optimizer.step()

    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs, _ = model(inputs)
            predictions = torch.sigmoid(outputs[:, -1])
            val_predictions.extend(predictions.numpy())
            val_labels.extend(labels.numpy())

    val_predictions = [1 if p > 0.5 else 0 for p in val_predictions]
    accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation accuracy: {accuracy:.2f}")

# Make predictions on new data
new_data = your_new_data  # Replace with your new, unseen data as a tensor
with torch.no_grad():
    outputs, _ = model(new_data)
    predictions = torch.sigmoid(outputs[:, -1])
    final_predictions = [1 if p > 0.5 else 0 for p in predictions.numpy()]

print("Predictions on new data:", final_predictions)




###########
# ROC
###########
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have a trained deep learning model and test data
# model = trained model
# X_test = test data
# y_test = test labels

# Predict the probability scores for the test data
y_pred = model.predict(X_test)

# Calculate the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
