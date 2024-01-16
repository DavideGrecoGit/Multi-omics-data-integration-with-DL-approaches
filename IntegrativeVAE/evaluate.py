import argparse
import pandas as pd
import numpy as np
import os
from classifiers import Benchmark_Classifier


def load_csv(path):
    return np.loadtxt(path, delimiter=",")


N_FOLDS = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-results_dir",
        help="Path to the directory of saved results",
        type=str,
        default="./results",
    )

    parser.add_argument(
        "-id",
        help="ID of the trained model to evaluate",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    fold_dir = "./data/5-fold_pam50stratified/"

    for dir in os.listdir(args.results_dir):
        for id in os.listdir(os.path.join(args.results_dir, dir)):
            if id != args.id:
                continue

            latent_dir = os.path.join(args.results_dir, dir, id)
            acc_scores = []

            for k in range(1, N_FOLDS + 1):
                print(f"=== FOLD {k} ===")

                train_embed = load_csv(
                    os.path.join(latent_dir, f"fold_{k}", "train_latent.csv")
                )
                test_embed = load_csv(
                    os.path.join(latent_dir, f"fold_{k}", "test_latent.csv")
                )

                train_gt = load_csv(
                    os.path.join(fold_dir, f"fold{k}", "train_pam50.csv")
                )
                test_gt = load_csv(os.path.join(fold_dir, f"fold{k}", "test_pam50.csv"))

                classifier = Benchmark_Classifier()
                accTrain, f1Train = classifier.train(train_embed, train_gt)
                accTest, f1Test = classifier.evaluate(test_embed, test_gt)

                print(f"\nTrain Acc: {accTrain}, Test Acc: {accTest}\n")

                acc_scores.append([*accTest, *f1Test])

            metrics = np.array(acc_scores).mean(axis=0)
            print(f"Mean Test metrics: {metrics}")
            columns = [
                "Acc_NB",
                "Acc_SVM",
                "Acc_RF",
                "f1_NB",
                "f1_SVM",
                "f1_RF",
            ]

            df = pd.DataFrame(
                [metrics],
                columns=columns,
            )
            df.to_csv(os.path.join(latent_dir, "metrics.csv"), index=False)
