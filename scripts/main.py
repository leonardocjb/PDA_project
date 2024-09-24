from trainTransformer import *
from trainVoiceNet import *
from bayesianOptimization import *


def main():
    # config = get_voiceNetConfig()
    # trainVoiceNet(config)
    # config = get_pitchTransformerConfig()
    # train_and_evaluate(config)

    tune_hyperparameters()


if __name__ == "__main__":
    main()
