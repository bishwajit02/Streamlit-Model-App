import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

seed = torch.manual_seed(42)

# Define Model
class Model(nn.Module):
    def __init__(self, in_features=5, h1=256, h2=512, h3=256, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.out = nn.Linear(h3, out_features)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        return self.out(x)



# Streamlit UI
st.title("Neural Network Trainer")
st.sidebar.header("Model Parameters")
h1 = st.sidebar.number_input("Neurons in Layer 1 (h1)", min_value=32, max_value=1024, value=256, step=32)
h2 = st.sidebar.number_input("Neurons in Layer 2 (h2)", min_value=32, max_value=1024, value=512, step=32)
h3 = st.sidebar.number_input("Neurons in Layer 3 (h3)", min_value=32, max_value=1024, value=256, step=32)
seed = st.sidebar.number_input("Input manual seed", min_value=0, max_value=500, value=42, step=10)
epochs = st.sidebar.number_input("Input Iteration Number", min_value=0, max_value=5000, value=500, step=50)


if st.button("Run Model"):
    # Load Data
    url = r"G:\Bishwajit\Dataset\data\Dataset\p.csv"  
    df = pd.read_csv(url)
    df.drop(columns=['target_name', 'target_classification'], inplace=True)
    X = df.drop('classification_id', axis=1).values
    y = df['classification_id'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    model = Model(in_features=X_train.shape[1], h1=h1, h2=h2, h3=h3, out_features=len(set(y)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 500
    losses = []
    
    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Accuracy Calculation
    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test)
    
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predicted_classes = torch.argmax(y_pred, dim=1)
            correct += (predicted_classes == y_batch).sum().item()
            total += y_batch.size(0)
    accuracy = correct / total

    # Display Results
    st.subheader("Results")
    st.write(f"Final Loss: {loss.item():.4f}")
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Plot Loss Curve
    fig, ax = plt.subplots()
    ax.plot(losses, label='Training Loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Training Losses")
    st.bar_chart(losses)

    actual_class_ids = []
    predicted_class_ids = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predicted_classes = torch.argmax(y_pred, dim=1)
            actual_class_ids.extend(y_batch.numpy())
            predicted_class_ids.extend(predicted_classes.numpy())
    
    # Display Sample Predictions
    st.subheader("Sample Predictions")
    prediction_data = {
        "Sample": [i+1 for i in range(10)],
        "Actual ID": actual_class_ids[:10],
        "Predicted ID": predicted_class_ids[:10]
    }
    prediction_df = pd.DataFrame(prediction_data)
    st.write(prediction_df)

    if st.button("Update Predictions"):
        # Re-run prediction logic and update the prediction_df
        actual_class_ids = []
        predicted_class_ids = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                predicted_classes = torch.argmax(y_pred, dim=1)
                actual_class_ids.extend(y_batch.numpy())
                predicted_class_ids.extend(predicted_classes.numpy())

        # Create a new prediction DataFrame
        updated_prediction_data = {
            "Sample": [i+1 for i in range(len(actual_class_ids))],
            "Actual ID": actual_class_ids,
            "Predicted ID": predicted_class_ids
        }
        updated_prediction_df = pd.DataFrame(updated_prediction_data)
        
        # Display the updated predictions
        st.subheader("Updated Predictions")
        st.write(updated_prediction_df.head(10))  # Show first 10 predictions

        # Display the updated prediction chart
        st.subheader("Updated Actual vs Predicted Class IDs")
        updated_prediction_df.set_index('Sample', inplace=True)
        st.line_chart(updated_prediction_df[['Actual ID', 'Predicted ID']])
