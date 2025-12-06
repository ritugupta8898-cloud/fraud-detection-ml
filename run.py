from datetime import datetime
from src.load_data import load_data
from src.preprocess import preprocess
from src.train import train_model
from src.train import train_model_svm
from src.evaluate import evaluate
from src.plots import plot_confusion_matrix, plot_roc_curve


print("Loading data...", datetime.now())
df = load_data()
'''
import matplotlib.pyplot as plt
import seaborn as sns

# Show all columns (just for confirmation)
print(df.columns)

# 1. Class Distribution (fraud vs non-fraud)
plt.figure(figsize=(6,4))
sns.countplot(x=df['Class'])
plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(12,5))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 3. Feature distribution histograms
df.hist(figsize=(14,10), bins=30)
plt.suptitle("Feature Distributions")
plt.show()

# 4. Outlier visualization
plt.figure(figsize=(12,5))
sns.boxplot(data=df.select_dtypes(include=['int64','float64']))
plt.xticks(rotation=90)
plt.title("Boxplot — Outlier Check")
plt.show()

'''
print("Preprocessing...", datetime.now())
X_train, X_test, y_train, y_test, scaler = preprocess(df)

print("Training...", datetime.now())
model = train_model(X_train, y_train, model_type="rf")
sample_size = 60000   # adjust if still slow (try 30k–80k)
X_train_small = X_train[:sample_size]
y_train_small = y_train[:sample_size] 
svm_model = train_model_svm(X_train_small, y_train_small)

print("Evaluating...", datetime.now())
evaluate(model, X_test, y_test)
preds = model.predict(X_test)
plot_confusion_matrix(y_test, preds)
plot_roc_curve(model, X_test, y_test)

print("Evaluating_for_SVM...", datetime.now())
evaluate(svm_model, X_test, y_test)
preds = svm_model.predict(X_test)
plot_confusion_matrix(y_test, preds)
plot_roc_curve(svm_model, X_test, y_test)


print("Done!", datetime.now())
