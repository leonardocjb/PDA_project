import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split
from pitchTransformer import *
from speech_dataset import *
from config import *
from ray import train, tune


def train_and_evaluate(tunerConfig):
    config = get_pitchTransformerConfig(tunerConfig)
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
    n_head = config["n_head"]
    batch_per_lr = config["batch_per_lr"]
    swa_start = config["swa_start"]
    swa_lr = config["swa_lr"]
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
        nhead=n_head,
        num_layers=6,
        output_dim=1,
        dropout=dropOut,
    )

    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_normal_(p)

    model.to(device)

    # criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    criterion = nn.SmoothL1Loss(beta=1)  # Using Huber loss in case of outliers
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
    best_val_mse = float("inf")

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=lr,
        epochs=swa_start,
        steps_per_epoch=int(len(train_loader) / batch_per_lr),
    )
    swa_start = config["swa_start"]
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)

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
            if epoch <= swa_start:
                scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = swa_model(inputs) if epoch > swa_start else model(inputs)
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
                "/cluster/projects/schwartzgroup/leo/PDA_project/debug/model.pt",
            )
            print(val_predictions)
            print("=================================")
            print(val_targets)
            exit(1)
        val_r2 = r2_score(val_targets, val_predictions)
        val_loss = val_loss / len(val_loader)

        if val_mse < best_val_mse:
            # Save the model if the current validation MSE is better than the best so far
            model_filename = get_weights_file_path(
                config=get_pitchTransformerConfig(tunerConfig), epoch=epoch
            )
            best_val_mse = val_mse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": swa_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict(),
                },
                model_filename,
            )

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}    "
            f"Validation Loss: {val_loss:.4f}, Validation MSE: {val_mse:.4f}, Validation R^2: {val_r2:.4f}"
        )

    # Testing
    model.eval()
    test_predictions = []
    test_targets = []
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = swa_model(inputs)
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(labels.cpu().numpy())
            test_loss += criterion(outputs, labels)

    test_predictions = np.concatenate(test_predictions).squeeze()
    test_targets = np.concatenate(test_targets).squeeze()
    test_mse = mean_squared_error(test_targets, test_predictions)

    test_r2 = r2_score(test_targets, test_predictions)
    test_acc = 1 - np.abs((test_predictions - test_targets)) / test_targets
    acc_mean = np.mean(test_acc)
    acc_std = np.std(test_acc)
    test_loss /= len(test_loader)
    test_loss = test_loss.cpu().numpy()
    gpe = np.sum(np.abs(test_predictions - test_targets) <= 0.2 * np.abs(test_targets))

    print(f"Testing Accuracy: {acc_mean} +- {acc_std}")
    print(
        f"Testing Loss {test_loss:.4f}, Testing MSE: {test_mse:.4f}, Testing R^2: {test_r2:.4f}"
    )

    print(f"gpe: {gpe}, test_loss: {test_loss}")
    return {"score": test_loss}
