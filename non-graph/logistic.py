import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from configs.base_config import BaseSettings as settings

# --- SMILES -> Morgan fingerprint ---
def smiles_to_morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# --- feature matrix ---
def build_feature_matrix(df, smiles_col="SMILES", target_col="target", n_bits=2048):
    X, y = [], []
    for i, row in df.iterrows():
        fp = smiles_to_morgan_fp(row[smiles_col], n_bits=n_bits)
        if fp is not None:
            X.append(fp)
            y.append(row[target_col])
        else:
            print(f"Skipping invalid SMILES at index {i}: {row[smiles_col]}")
    return np.array(X), np.array(y)


def evaluate_model(model, X, y, name="Dataset"):
    y_pred = model.predict(X)
    y_proba = None
    try:
        y_proba = model.predict_proba(X)
    except:
        pass

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    auc = None
    if y_proba is not None:
        try:
            if len(np.unique(y)) == 2:
                auc = roc_auc_score(y, y_proba[:, 1])
            else:
                auc = roc_auc_score(y, y_proba, multi_class="ovr")
        except Exception as e:
            print(f"AUC could not be computed for {name}: {e}")

    print(f"\n{name} Results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    if auc is not None:
        print(f"  AUC:       {auc:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


if __name__ == "__main__":

    train_df = pd.read_csv(settings.TRAIN_DATA)
    val_df = pd.read_csv(settings.VAL_DATA)
    test_df = pd.read_csv(settings.TEST_DATA)

    X_train, y_train = build_feature_matrix(train_df)
    X_val, y_val = build_feature_matrix(val_df)
    X_test, y_test = build_feature_matrix(test_df)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])
    clf.fit(X_train, y_train)

    train_metrics = evaluate_model(clf, X_train, y_train, name="Train")
    val_metrics = evaluate_model(clf, X_val, y_val, name="Validation")
    test_metrics = evaluate_model(clf, X_test, y_test, name="Test")
