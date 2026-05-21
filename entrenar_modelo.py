"""
Entrena un clasificador con el dataset de señas.
Usa RandomForest de scikit-learn. Guarda modelo.pkl para el predictor.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

ARCHIVO_DATASET = "dataset.csv"
ARCHIVO_MODELO = "modelo.pkl"


def main():
    if not os.path.exists(ARCHIVO_DATASET):
        print(f"Error: No existe {ARCHIVO_DATASET}")
        print("Ejecuta primero: python recolectar_dataset.py")
        return

    df = pd.read_csv(ARCHIVO_DATASET)
    X = df.drop(columns=["etiqueta"])
    y = df["etiqueta"]

    if len(df) < 5:
        print(f"Error: Pocas muestras ({len(df)}). Recolecta al menos 5 por etiqueta.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n--- Resultados ---")
    print(f"Exactitud: {acc:.2%}")
    print("\nReporte por clase:")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, ARCHIVO_MODELO)
    print(f"\nModelo guardado en {ARCHIVO_MODELO}")


if __name__ == "__main__":
    main()
