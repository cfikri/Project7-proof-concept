import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# La fonction *valeurs_manquantes* ci-dessous permet de déterminer le nombre et le pourcentage de valeurs manquantes (à 0.1% près) de chaque features d'un dataset.

def valeurs_manquantes(DataFrame):
    effectif = DataFrame.isna().sum()
    taux = DataFrame.isna().mean().round(3)*100
    result = pd.DataFrame({'effectif' : effectif, 'taux' : taux})
    return result.loc[result.effectif !=0, :] 

# La fonction *stats* ci-dessous prend en argument un DataFrame et renvoie un tableau contenant les principaux indicateurs statistiques de ses variables (effectif, moyenne, écart-type, médiane, quartiles, min et max).

def stats(DataFrame):
    return DataFrame.describe().round(3).T


# La fonction *stats_extend* ci-dessous vise à présenter sous-forme de tableau les principaux indicateurs statistiques d'une DataFrame :
# - Les indicateurs de tendance centrale : moyenne et médiane ;
# - Les indicateurs de dispersion : étendue ,écart-type, quartiles et écart-interquartile ;
# - Les indicateurs de forme : skewness (asymétrie) et kurtosis (aplatissement).

def stats_extend(DataFrame):
    result = stats(DataFrame)
    quantitatif = DataFrame.select_dtypes(include=['int', 'float'])
    result.rename(columns = {'25%':'Q1', '50%':'med', '75%':'Q3' }, inplace=True)
    del result['count']
    result['etendue'] = result['max'] - result['min']
    result['IQR'] = result['Q3'] - result['Q1']
    result['skew'] = quantitatif.skew()
    result['kurtosis'] = quantitatif.kurtosis()
    return result

# Créer des sous-séquences
def create_subsequences(serie):
    timelist = range(0, 192, 32)
    subsequences = []
    for i in timelist:
        subsequences.append(serie[i:i+32])
    subsequences = np.transpose(subsequences, (1, 0))        
    return np.array(subsequences)