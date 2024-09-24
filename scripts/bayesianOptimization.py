from ray import tune
from ray.tune.search.hebo import HEBOSearch
from trainTransformer import *


def tune_hyperparameters():
    # Define the search space
    searchSpace = {
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "d_model": tune.choice([32, 64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.9),
        "weightDecay": tune.loguniform(1e-6, 1e-2),
        "n_head": tune.choice([2, 4, 8]),
        "batch_per_lr": tune.choice([1, 2, 4]),
        "swa_start": tune.choice([30, 50, 70]),
        "swa_lr": tune.loguniform(1e-4, 1e-2),
    }
    hebo = HEBOSearch(metric="score", mode="min")
    train_model = tune.with_resources(train_and_evaluate, {"gpu": 1})
    tuner = tune.Tuner(
        train_model,
        tune_config=tune.TuneConfig(search_alg=hebo, num_samples=4),
        param_space=searchSpace,
    )
    result = tuner.fit()
    bestResult = result.get_best_result(metric="score", mode="min")
    print(bestResult.metrics)
    print(bestResult.metrics_dataframe)
    for i, result in enumerate(result):
        if result.error:
            print(f"Trial #{i} had an error:", result.error)
            continue

        print(
            f"Trial #{i} finished successfully with a score of:",
            result.metrics["score"],
        )

    # analysis = tune.run(train_and_evaluate, config=searchSpace, resources_per_trial={'gpu': 1})
