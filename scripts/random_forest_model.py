from dados_treino import train_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def random_forest():
    # Dados de treino
    df, x, y, x_train, x_test, y_train, y_test = train_data()

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    print(classification_report(y_test, y_pred_rf))


if __name__ == "__main__":
    random_forest()