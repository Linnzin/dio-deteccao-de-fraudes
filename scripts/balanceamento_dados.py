import pandas as pd

from dados_treino import train_data
from imblearn.over_sampling import SMOTE


def data_balancing():
    # Dados de treino
    df, x, y, x_train, x_test, y_train, y_test = train_data()

    # Balanceamento por Undersampling
    fraudes = df[df["Class"] == 1]
    legitimas = df[df["Class"] == 0].sample(len(fraudes), random_state=42)
    df_under = pd.concat([fraudes, legitimas])

    # Balanceamento por Oversampling
    smote = SMOTE()
    x_res, y_res = smote.fit_resample(x, y)


if __name__ == "__main__":
    data_balancing()