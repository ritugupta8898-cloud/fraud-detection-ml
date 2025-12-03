import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    from imblearn.over_sampling import RandomOverSampler

    oversampler = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = oversampler.fit_resample(X_train, y_train)

    

    return X_train_res, X_test, y_train_res, y_test, scaler
