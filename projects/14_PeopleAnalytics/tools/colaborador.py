# Imports
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier


#Criação da classe
class Colaborador:

    def __init__(self, nivel_satisfacao, tempo_empresa, numero_projetos, horas_medias_por_mes, ultima_avaliacao):
        self.array = np.array([nivel_satisfacao, tempo_empresa, numero_projetos, horas_medias_por_mes, ultima_avaliacao])

    def predict(self):
        model = joblib.load('notebook/modelo/modelo-20220725-164032.pkl')
        pred = model.predict(self.array.reshape(1, -1))
        return pred
        