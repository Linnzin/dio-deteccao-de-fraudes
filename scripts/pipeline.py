from dados_treino import train_data
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

def main():
    # Dados de treino
    # O parâmetro 'scaler_transform=False' evita redundância no scaler. 
    df, x, y, x_train, x_test, y_train, y_test = train_data(scaler_transform=False)

    # Utilização do pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()