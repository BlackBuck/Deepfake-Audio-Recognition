import torch
import torch.nn as nn
from preprocessing import train_dataloader, test_dataloader
from torchvision.models import mobilenet_v2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetV2, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        x = self.mobilenet(x)
        return x
    
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
    # Example usage
    num_classes = 1
    mobilenet = MobileNetV2(num_classes)

    # Example training loop
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mobilenet.parameters(), lr=1e-3)

    # Assuming train_loader and test_loader are defined and device is set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    mobilenet.to(device)
    train(mobilenet, train_dataloader, device, criterion, optimizer)
    print(mobilenet)
    # num_epochs = 100
    # for epoch in range(num_epochs):
    #     mobilenet.train()
    #     for i, data in enumerate(train_dataloader, 0):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = mobilenet(inputs.float())
    #         # print(outputs)
    #         # print(labels)
    #         loss = criterion(outputs, labels.float())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    torch.save(mobilenet.state_dict(), 'mobilenet_trained2.h5')
    # mobilenet.load_state_dict(torch.load('./mobilenet_trained2.h5'))
    evaluate_model(mobilenet, test_dataloader, device)
    # mobilenet.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for inputs, labels in test_dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = mobilenet(inputs)
    #         _, predicted = torch.max(outputs.data, 1)
    #         _, labels = torch.max(labels, 1)
    #         print(predicted)
    #         print(labels)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # print(f'Accuracy: {100 * correct / total:.2f}%')
    