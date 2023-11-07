from trainTransformer import *
from trainVoiceNet import *

def main():
    # config = get_voiceNetConfig()
    # trainVoiceNet(config)
    config = get_pitchTransformerConfig()
    train_and_evaluate(config)

if __name__ == "__main__":
    main()