import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


class Benchmark_Classifier:
    def __init__(self):
        super().__init__()
        self.classifiers = [
            GaussianNB(),
            SVC(
                C=1.5,
                kernel="rbf",
                random_state=SEED,
                gamma="auto",
            ),
            RandomForestClassifier(
                n_estimators=50,
                random_state=SEED,
                max_features=0.5,
            ),
        ]

    def train(self, latent, gt):
        # print(latent_train.shape)
        print("Training Classifiers")

        for cls in self.classifiers:
            cls.fit(latent, gt)

        return self.evaluate(latent, gt)

    def evaluate(self, latent, gt):
        acc_scores = []
        f1_scores = []
        for cls in self.classifiers:
            predictions = cls.predict(latent)
            # print(predictions.shape)
            acc_scores.append(accuracy_score(gt, predictions))
            f1_scores.append(f1_score(gt, predictions, average="macro"))

        return acc_scores, f1_scores
