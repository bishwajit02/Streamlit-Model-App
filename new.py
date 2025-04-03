import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
from io import StringIO

# Set a default seed for initial UI load
torch.manual_seed(42)

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
h1 = st.sidebar.number_input("Neurons in Layer 1 (h1)", min_value=32, max_value=1024, value=256, step=32, key="h1")
h2 = st.sidebar.number_input("Neurons in Layer 2 (h2)", min_value=32, max_value=1024, value=512, step=32, key="h2")
h3 = st.sidebar.number_input("Neurons in Layer 3 (h3)", min_value=32, max_value=1024, value=256, step=32, key="h3")
manual_seed = st.sidebar.number_input("Input manual seed", min_value=0, max_value=500, value=42, step=10, key="seed")
epochs = st.sidebar.number_input("Input Iteration Number", min_value=0, max_value=5000, value=500, step=50, key="epochs")

raw_url = st.text_input("Enter the URL of Dataset: ", key="raw_url")
data_df = None
classification_map = {}

if raw_url:
    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        csv_data = StringIO(response.text)
        data_df = pd.read_csv(csv_data)
        
        classification_map = data_df.set_index("classification_id")["target_classification"].to_dict()
        
        data_df.drop(columns=["target_name", "target_classification"], inplace=True, errors='ignore')
        st.subheader("Cleaned Data Preview")
        st.dataframe(data_df)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

if st.button("Run Model") and data_df is not None:
    # Training process
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)

    X = data_df.drop("classification_id", axis=1).values
    y = data_df["classification_id"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=manual_seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    model = Model(in_features=X_train.shape[1], h1=h1, h2=h2, h3=h3, out_features=len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    accuracies = []
    progress_bar = st.progress(0)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X_train)
        
        # Compute loss
        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        
        # Compute accuracy
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y_train).sum().item()
        accuracy = correct / y_train.size(0)
        accuracies.append(accuracy)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        progress_bar.progress((epoch + 1) / epochs)
    
    # Display the loss vs epoch graph using Matplotlib
    st.subheader("Training Loss vs Epochs")
    fig_loss = plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), losses, label='Loss', color='red')
    plt.title('Training Loss vs Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    st.pyplot(fig_loss)
    
    # Display the accuracy vs epoch graph using Streamlit line chart for improved design
    st.subheader("Training Accuracy vs Epochs")
    accuracy_df = pd.DataFrame({'Epochs': range(epochs), 'Accuracy': accuracies})
    st.line_chart(accuracy_df.set_index('Epochs'), use_container_width=True)
    
    # Model Testing (Accuracy and Loss)
    st.subheader("Model Accuracy on Test Data")
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        _, y_test_pred_class = torch.max(y_test_pred, 1)
        correct = (y_test_pred_class == y_test).sum().item()
        accuracy = correct / y_test.size(0)
        st.write(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Side-by-side comparison: Actual vs Predicted Classes
    st.subheader("Side-by-Side Comparison: Actual vs Predicted Class IDs")
    comparison_df = pd.DataFrame({
        'Actual ID': y_test.numpy(),
        'Predicted ID': y_test_pred_class.numpy(),
        'Actual Classification': [classification_map.get(i, "Unknown") for i in y_test.numpy()],
        'Predicted Classification': [classification_map.get(i, "Unknown") for i in y_test_pred_class.numpy()]
    })
    
    st.dataframe(comparison_df)
