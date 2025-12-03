from datetime import datetime
from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.evaluate import evaluate

print("Loading data...", datetime.now())
df = load_data()

print("Preprocessing...", datetime.now())
X_train, X_test, y_train, y_test, scaler = preprocess(df)

print("Training...", datetime.now())
model = train_model(X_train, y_train, model_type="rf")

print("Evaluating...", datetime.now())
evaluate(model, X_test, y_test)

print("Done!", datetime.now())
