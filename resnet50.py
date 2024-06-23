import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from preprocessing import train_dataloader, test_dataloader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(model, dataloader, device, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float()) 
            # print(outputs)
            #outputs are ranging from 0 to some positive number, so I change it 
            # preds = torch.tensor([[1 if outputs[i][0] >= 1 else 0] for i in range(len(outputs))]).to(device)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.cpu().apply_(lambda x : 1 if x >= 0 else 0)
            y_true.extend(y.to(torch.device("cpu")))
            y_pred.extend(preds.float().to(torch.device("cpu")))
    print("Y_TRUE SHAPE ", y_true)
    print("Y_PRED SHAPE ", y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')


if __name__ == "__main__":
    
    net = resnet50(pretrained=False)
    net.fc = nn.Linear(2048, 1)
    # print(net)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.to(device)
    # train(net, train_dataloader, device, criterion, optimizer, epochs=100)
    # torch.save(net.state_dict(), 'resnet50_trained.h5')
    net.load_state_dict(torch.load("resnet50_trained.h5"))

    evaluate_model(net, test_dataloader, device)

