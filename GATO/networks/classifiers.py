import torch
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils.train_val_test import setup_seed

SEED = setup_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Benchmark_Classifier:
    def __init__(self, model):
        super().__init__()
        self.model = model
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

    def train(self, dataloader):
        latent_train, gt_train = self.model.get_latent_space(dataloader)
        print("Training Classifiers")

        for cls in self.classifiers:
            cls.fit(latent_train, gt_train)

        return self.evaluate(dataloader)

    def evaluate(self, dataloader):
        latent, gt = self.model.get_latent_space(dataloader)
        acc_scores = []
        for cls in self.classifiers:
            predictions = cls.predict(latent)
            acc_scores.append(accuracy_score(gt, predictions))

        return acc_scores
