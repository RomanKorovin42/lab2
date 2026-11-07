"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    
    tableInfo = pd.read_csv(data)
    df = pd.DataFrame(tableInfo)

    # Кол-во людей старше 30
    def getAgeCount():
        ageStat = len(df[df['Age'] > 30])
        print(f'Количество людей старше 30: {ageStat}\n')

        return ageStat

    # Средний возраст для каждого значения Pclass
    def getAvgAgeByPclass():
        pclassDict = {}
        resultDict = {}

        # Удаляем строки, где возраст не указан
        dfCleaned = df.dropna(subset=['Age'])

        for i in dfCleaned['Pclass'].to_list():
            if i in pclassDict.keys():
                pclassDict[i] += 1
            else:
                pclassDict[i] = 1

        pclassDict = dict(sorted(pclassDict.items()))

        for i in pclassDict.keys():
            dfFiltered = df[df['Pclass'] == i]
            resultDict[i] = round(dfFiltered['Age'].sum() / pclassDict[i])
            print(f'Средний возраст для Pclass {i}:{resultDict[i]}\n')

        return resultDict

    # Пять наибольших значений Fare в порядке убывания.
    def getTopFare():
        topFare = df.sort_values('Fare', ascending=False)
        resultList = topFare['Fare'].to_list()[:5]
        print(f'Пять наибошльших значений Fare: {resultList}\n')

        return resultList

    # Доля выживших для каждого Pclass
    def getSurvivoursPart():
        pclassDict = {}
        pclassAvgDict = {}
        dfSurvivours = df[df['Survived'] == 1]

        for i in dfSurvivours['Pclass'].to_list():
            if i in pclassDict.keys():
                pclassDict[i] += 1
            else:
                pclassDict[i] = 1

        pclassDict = dict(sorted(pclassDict.items()))

        dfCount = len(df['PassengerId'].to_list())

        for i in pclassDict.keys():
            pclassAvgDict[i] = round((pclassDict[i] / dfCount), 2)
            print(f'Выжившие для Pclass {i}: {pclassAvgDict[i]}\n')

        return pclassAvgDict

    # Количество пропусков в каждом столбце
    def getEmptyCells():
        columnsDict = {}

        for column in df:
            columnsDict[column] = df[column].isnull().sum()

        for i in columnsDict.keys():
            print(f'Пропуски в столбце {i}:{columnsDict[i]}\n')

        return columnsDict

    emptyCells = getEmptyCells()
    ageOver30Count = getAgeCount()
    avgAgeByPclass = getAvgAgeByPclass()
    survivoursPart = getSurvivoursPart()
    topFare = getTopFare()

    return TitanicSummary(len(df), emptyCells, ageOver30Count, avgAgeByPclass, survivoursPart, topFare)


analyze_titanic('titanic.csv')
