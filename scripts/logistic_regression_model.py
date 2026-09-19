import matplotlib.pyplot as plt
import os
import warnings

from dados_treino import train_data
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve


def logistic_regression():
    # Dados de treino
    df, x, y, x_train, x_test, y_train, y_test = train_data()

    # Regressão Logística
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("\n-----------------------------------------\n")
    print(classification_report(y_test, y_pred))

    # Curva ROC
    y_probs = model.predict_proba(x_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    os.makedirs("analises_graficas", exist_ok=True)
    plt.savefig("analises_graficas/roc_curve.png")
    plt.clf()
    print("\n-----------------------------------------\n")
    print("AUC:", roc_auc_score(y_test, y_probs))

    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    os.makedirs("analises_graficas", exist_ok=True)
    plt.savefig("analises_graficas/precision_recall_curve.png")
    plt.clf()

    # Alterando o threshold para encontrar mais fraudes
    threshold = 0.3
    y_pred_custom = (y_probs > threshold).astype(int)
    print("\n-----------------------------------------\n")
    print(classification_report(y_test, y_pred_custom))

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    logistic_regression()