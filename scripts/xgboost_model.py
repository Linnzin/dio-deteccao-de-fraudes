import matplotlib.pyplot as plt
import os

from dados_treino import train_data
from sklearn.metrics import classification_report
from xgboost import XGBClassifier 


def xgboost_model():
    # Dados de treino
    df, x, y, x_train, x_test, y_train, y_test = train_data()

    # Extreme Gradient Boosting
    xgb = XGBClassifier(
        scale_pos_weight=10,
        eval_metric="logloss"
    )
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    print(classification_report(y_test, y_pred_xgb))

    # Gráfico: Importância das Variáveis
    importances = xgb.feature_importances_
    plt.bar(range(len(importances)), importances)
    plt.title("Importância das Variáveis")
    os.makedirs("analises_graficas", exist_ok=True)
    plt.savefig("analises_graficas/importances.png")


if __name__ == "__main__":
    xgboost_model()