# logistic regression model
from sklearn.linear_model import LogisticRegression

class Model:
    def __init__(self, random_state=0):
        self.model = LogisticRegression(random_state=random_state)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)