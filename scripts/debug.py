import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, random_split

from pitchTransformer import *
from speech_dataset import *
from config import *

config = get_pitchTransformerConfig()
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
input_size = 384

model = audioPitchTransformer(
    input_size=input_size,
    d_model=d_model,
    nhead=4,
    num_layers=6,
    output_dim=1,
    dropout=dropOut,
)

saved_data = torch.load(
    "/cluster/projects/schwartzgroup/leo/PDA_project/audioTransformer_weights/audioPitchTransformer_87.pt"
)

model.load_state_dict(saved_data.get("model_state_dict"))
model.to(device)
dataset = audioPitchTranformerData(zeroRate=zeroRate)
trainData, valData, testData = random_split(dataset, trainTestRatio)
val_loader = DataLoader(
    dataset=valData, batch_size=batch_size, shuffle=False, num_workers=4
)
train_loader = DataLoader(
    dataset=trainData, batch_size=batch_size, shuffle=True, num_workers=4
)
criterion = nn.SmoothL1Loss(beta=5)
model.train()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
optimizer.load_state_dict(saved_data.get("optimizer_state_dict"))

train_loss = 0.0
for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
train_loss /= len(train_loader)

# Validation
model.eval()


for i in range(100):
    val_predictions = []
    val_targets = []
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_predictions.append(outputs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_predictions = np.concatenate((val_predictions)).squeeze()
    val_targets = np.concatenate(val_targets).squeeze()

    val_mse = mean_squared_error(val_targets, val_predictions)

    val_r2 = r2_score(val_targets, val_predictions)
    print(i, val_mse)
