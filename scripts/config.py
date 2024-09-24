from pathlib import Path


def get_voiceNetConfig():
    return {
        "batch_size": 64,
        "epochs": 30,
        "lr": 0.0001,
        "trainTestRatio": [0.7, 0.2, 0.1],
        "num_folds": 5,
        "model_folder": "weights",
        "model_name": "voiceNet",
        "experiment_name": "runs/voiceNet",
    }


def get_pitchTransformerConfig(tunerConfig: dict):
    return {
        "batch_size": tunerConfig["batch_size"],
        "epochs": 10,
        "n_head": tunerConfig["n_head"],
        "batch_per_lr": tunerConfig["batch_per_lr"],
        "lr": tunerConfig["lr"],
        "d_model": tunerConfig["d_model"],
        "zeroRate": 0.00,
        "dropOut": tunerConfig["dropout"],
        "weightDecay": tunerConfig["weightDecay"],
        "swa_start": tunerConfig["swa_start"],
        "swa_lr": tunerConfig["swa_lr"],
        "trainTestRatio": [0.7, 0.2, 0.1],
        "model_folder": "audioTransformer_weights",
        "model_name": "audioPitchTransformer",
        "experiment_name": "runs/audioPitchTransformer",
    }


def get_weights_file_path(config, epoch: str):
    HomeFolder = "/cluster/projects/schwartzgroup/leo/PDA_project/"
    model_folder = config["model_folder"]
    model_name = config["model_name"]
    file_name = f"{model_name}_{epoch}.pt"
    return HomeFolder + model_folder + "/" + file_name
