import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

SEED = 42


class Benchmark_Classifier:
    def __init__(self, NB=True, SVM=True, RF=True):
        super().__init__()

        self.classifiers = []
        if NB:
            self.classifiers.append(GaussianNB())
        if SVM:
            self.classifiers.append(
                SVC(
                    C=1.5,
                    kernel="rbf",
                    random_state=SEED,
                    gamma="auto",
                )
            )
        if RF:
            self.classifiers.append(
                RandomForestClassifier(
                    n_estimators=50,
                    random_state=SEED,
                    max_features=0.5,
                )
            )

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
