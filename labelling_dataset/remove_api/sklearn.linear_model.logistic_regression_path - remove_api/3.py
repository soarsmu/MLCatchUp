import pandas as pd

base = pd.read_csv('../csv/risco-credito2.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelenconder = LabelEncoder()
previsores[:, 0] = labelenconder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelenconder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelenconder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelenconder.fit_transform(previsores[:, 3])

from sklearn.linear_model import logistic_regression_path
classificador = logistic_regression_path(X=previsores, y=classe)

# historia boa, divida alta, garantias nenhuma, renda > 35
# historia ruim, divida alta, garantias adequada, renda < 35
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])

print(classificador)
# print(classificador.class_count_)
# print(classificador.class_prior_)
# print(resultado)