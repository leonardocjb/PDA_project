import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split

from pitchTransformer import *
from speech_dataset import *
from config import *


def train_and_evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device} as device")

    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    trainTestRatio = config["trainTestRatio"]
    d_model = config["d_model"]
    zeroRate = config["zeroRate"]
    weightDecay = config["weightDecay"]
    dropOut = config["dropOut"]
    dataset = audioPitchTranformerData(zeroRate=zeroRate)
    trainData, valData, testData = random_split(dataset, trainTestRatio)

    train_loader = DataLoader(
        dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset=valData, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        dataset=testData, batch_size=batch_size, shuffle=False, num_workers=4
    )

    input_size = dataset[0][0].shape[1]
    model = audioPitchTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=4,
        num_layers=6,
        output_dim=1,
        dropout=dropOut,
    )

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    model.to(device)

    m = 0
    for p in model.parameters():
        m += 1
    print(m)

    # criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    criterion = nn.SmoothL1Loss(beta=5)  # Using Huber loss in case of outliers
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    best_val_mse = float("inf")
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=0.9, last_epoch=-1
    )

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sched.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
                val_loss += criterion(outputs, labels).item()

        val_predictions = np.concatenate((val_predictions)).squeeze()
        val_targets = np.concatenate(val_targets).squeeze()
        try:
            val_mse = mean_squared_error(val_targets, val_predictions)
        except:
            torch.save(
                {"model": model, "data": val_loader},
                "/cluster/projects/schwartzgroup/leo/PDA_project/debug",
            )
            print(val_predictions)
            print("=================================")
            print(val_targets)
            exit(1)
        val_r2 = r2_score(val_targets, val_predictions)
        print(val_loss / len(val_loader))

        if val_mse < best_val_mse:
            # Save the model if the current validation MSE is better than the best so far
            model_filename = get_weights_file_path(
                config=get_pitchTransformerConfig(), epoch=epoch
            )
            best_val_mse = val_mse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": sched.state_dict(),
                },
                model_filename,
            )

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}    "
            f"Validation MSE: {val_mse:.4f}, Validation R^2: {val_r2:.4f}"
        )

    # Testing
    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(labels.cpu().numpy())

    test_predictions = np.concatenate(test_predictions).squeeze()
    test_targets = np.concatenate(test_targets).squeeze()
    test_mse = mean_squared_error(test_targets, test_predictions)

    test_r2 = r2_score(test_targets, test_predictions)
    test_acc = 1 - np.abs((test_predictions - test_targets)) / test_targets
    acc_mean = np.mean(test_acc)
    acc_std = np.std(test_acc)
    print(f"Testing Accuracy: {acc_mean} +- {acc_std}")
    print(f"Testing MSE: {test_mse:.4f}, Testing R^2: {test_r2:.4f}")

    return test_mse, test_r2
