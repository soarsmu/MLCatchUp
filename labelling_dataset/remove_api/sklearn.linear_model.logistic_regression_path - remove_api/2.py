import pandas as pd

base = pd.read_csv('../csv/credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.linear_model import logistic_regression_path
classificador = logistic_regression_path(X=previsores_teste, y=classe_treinamento)
# classificador.fit(previsores_treinamento, classe_treinamento)
# previsoes = classificador.predict(previsores_teste)

# from sklearn.metrics import confusion_matrix, accuracy_score
# precisao = accuracy_score(classe_teste, previsoes)
# matriz = confusion_matrix(classe_teste, previsoes)
#
# print(precisao)
# print(matriz)