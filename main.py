import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample text data and labels
data = {"text": ["This is a positive sentence.", "This is a negative sentence.", "Another positive example."],
        "label": [1, 0, 1]}
df = pd.DataFrame(data)

# Tokenize the text data using Keras
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(sequences)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df["label"], test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.LongTensor(X_train)
y_train = torch.FloatTensor(y_train.values)
X_test = torch.LongTensor(X_test)
y_test = torch.FloatTensor(y_test.values)

# Define a simple neural network using PyTorch
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Create a data loader for PyTorch
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Initialize the model and optimizer
input_dim = X_train.shape[1]
model = SimpleNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
losses = []  # To store training losses for visualization
for epoch in range(epochs):
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

# Visualize the training loss
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.show()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.float())
    test_outputs = (test_outputs > 0.5).float()
    accuracy = torch.sum(test_outputs == y_test).item() / len(y_test)
    print("Test Accuracy: {:.2f}%".format(accuracy))
