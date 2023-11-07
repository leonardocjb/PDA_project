import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import math
from voiceNet import *

def voiceNetModelEvaluation(pred, lab):
    predicted = pred.cpu().numpy()
    label = lab.cpu().numpy()
    cm = confusion_matrix(label, predicted)
    score = (cm[0, 0] + cm[1, 1]) / (cm[0, 0] + cm[1, 0] + cm[1, 1])
    return score


def trainVoiceNet(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} as device")

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    trainTestRatio = config["trainTestRatio"]

    dataset = voicedData()
    
    trainData, valData, testData = random_split(dataset, trainTestRatio)

    train_loader = DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=valData, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=testData, batch_size=batch_size, shuffle=False, num_workers=4)

    input_size = dataset[0][0].shape[0]
    # model = VoiceClassificationModel(input_size)
    model = voiceNet(input_size)

    # Move the model to the selected device
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))

    # Training loop with validation
    for epoch in range(epochs):
        model.train()
        # torch.autograd.set_detect_anomaly(True)
        # Training
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            inputs = torch.reshape(inputs, (-1, 1, 384))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sched.step()

        # Validation
        model.eval()
        total_correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = torch.reshape(inputs, (-1, 1, 384))
                outputs = model(inputs)
                predicted = (outputs >= 0.5)
                total_correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = total_correct_val / total_val

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}, Validation Accuracy: {val_accuracy * 100:.2f}%')

    # Evaluate the model on the test set
    model.eval()
    total_correct_test = 0
    total_test = 0
    TP_TN_FN = 0
    finalCM = np.zeros((2, 2))

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.reshape(inputs, (-1, 1, 384))
            outputs = model(inputs)
            predicted = (outputs >= 0.5)
            total_correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)
            cm = confusion_matrix(labels.cpu().numpy().flatten(), predicted.cpu().numpy().flatten())
            finalCM += cm
    
    TP_TN_FN = finalCM[1, 1] + finalCM[0, 0] + finalCM[1, 0]
    test_criterion = total_correct_test / TP_TN_FN
    test_accuracy = total_correct_test / total_test
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
    print(f'Special Criterion: {test_criterion* 100:.2f}%')
    print(finalCM)
    return test_criterion





