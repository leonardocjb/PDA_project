from pathlib import Path

def get_voiceNetConfig():
    return{
        "batch_size" : 64, 
        "epochs" : 30, 
        "lr" : 0.0001,
        "trainTestRatio" : [0.7, 0.2, 0.1],
        "num_folds" : 5,
        "model_folder" : "weights",
        "model_name": "voiceNet",
        "experiment_name" : "runs/voiceNet"
    }
def get_pitchTransformerConfig():
    return{
        "batch_size" : 32, 
        "epochs" : 100, 
        "lr" : 0.0001,
        "d_model" : 64,
        "zeroRate" : 0.00,
        "dropOut" : 0.5,
        "weightDecay": 0.1,
        "trainTestRatio" : [0.7, 0.2, 0.1],
        "model_folder" : "audioTransformer_weights",
        "model_name": "audioPitchTransformer",
        "experiment_name" : "runs/audioPitchTransformer"
    }

def get_weights_file_path(config, epoch : str):
    model_folder = config["model_folder"]
    model_name = config["model_name"]
    file_name = f"{model_name}_{epoch}.pt"
    return str(Path(".")/model_folder/file_name)