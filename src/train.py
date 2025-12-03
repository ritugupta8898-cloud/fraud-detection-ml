from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, model_type="logistic"):

    if model_type == "logistic":
        model = LogisticRegression(max_iter=500)

    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            n_jobs=-1
        )

    else:
        raise ValueError("Unknown model type")

    print(f"Training {model_type}...")
    model.fit(X_train, y_train)

    return model
