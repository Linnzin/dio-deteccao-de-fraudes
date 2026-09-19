import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def train_data(output_terminal: bool = False, scaler_transform: bool = True):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)

    if output_terminal:
        print("\n-----------------------------------------\n")

        print(df.head())

        # INTERPRETAÇÕES DO DATASET
        # 
        # coluna "time":        momento da transação
        # coluna "V1"-"V28":    variáveis transformadas
        # coluna "Amount":      valor da transação
        # coluna "Class":       indicação de transação fraudulenta (0 = legítima; 1 = fraudulenta) 

        print("\n-----------------------------------------\n")

        print(df["Class"].value_counts(normalize=True))

        # Se trata de um dataset desbalanceado!
        # 99% dos dados são de transações legítimas.
        # Um modelo treinado com esse dataset é ótimo para detectar transações legítimas, mas péssimo para detectar fraudes.

        print("\n-----------------------------------------\n")

    # Feature Engineering

    # Transformação logarítmica na coluna Amount para normalizar os dados.
    df["amount_log"] = np.log1p(df["Amount"])

    # Padronização de escala dos valores
    if scaler_transform:
        scaler = StandardScaler()
        df["amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    if output_terminal:
        print("Transformação logarítmica\n")
        print(df["amount_log"])
        print("\nPadronização de escala\n")
        print(df["amount_scaled"])
        print("\n-----------------------------------------\n")

    # Gerando dados de treino

    x = df.drop("Class", axis=1)
    y = df["Class"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=0.3, random_state=42
    )

    return df, x, y, x_train, x_test, y_train, y_test

if __name__ == "__main__":
    train_data(True)