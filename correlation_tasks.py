"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd
import numpy as np


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data, delimiter=None).dropna()

    # Считает корреляцию у женщин если genderValue равен True, иначе у мужчин
    def getPearsonCor(genderValue):
        gender = 'Female' if genderValue else 'Male'
        dfFiltered = df[df['Gender'] == gender]

        rFSIQ = np.corrcoef(dfFiltered['FSIQ'], dfFiltered['MRI_Count'])[0, 1]
        rVIQ = np.corrcoef(dfFiltered['VIQ'], dfFiltered['MRI_Count'])[0, 1]
        rPIQ = np.corrcoef(dfFiltered['PIQ'], dfFiltered['MRI_Count'])[0, 1]
        rWeight = np.corrcoef(dfFiltered['Weight'], dfFiltered['MRI_Count'])[0, 1]
        rHeight = np.corrcoef(dfFiltered['Height'], dfFiltered['MRI_Count'])[0, 1]

        result = {'Gender': gender,
                  'rFSIQ': rFSIQ,
                  'rVIQ': rVIQ,
                  'rPIQ': rPIQ,
                  'rWeight': rWeight,
                  'rHeight': rHeight}

        print(f'Корреляция среди {"женщин" if genderValue else "мужчин"} \nrFSIQ: {rFSIQ}\nrVIQ: {rVIQ}\nrPIQ: {rPIQ}\nrWeight: {rWeight}\nrHeight: {rHeight}\n')

        return result

    # Возвращает название признака, у которого модуль корреляции с MRI_Count максимален среди результатов обеих групп.
    def strongest_mri_feature(table1, table2):
        resName = ''
        resValue = 0

        for column in table1.keys():
            m = table1[column]
            if isinstance(m, float):
                if abs(m) > resValue:
                    resName = column
                    resValue = m

        for column in table2.keys():
            m = table2[column]
            if isinstance(m, float):
                if abs(m) > resValue:
                    resName = column
                    resValue = m

        print(f'Сильнейший параметр корреляции у обеих групп: {resName}')

        return resName


    maleTable = getPearsonCor(False)
    femaleTable  = getPearsonCor(True)
    maxMRI = strongest_mri_feature(maleTable, femaleTable)

    return BrainCorrelationSummary(len(maleTable), len(femaleTable), femaleTable, maleTable, maxMRI)


analyze_brain_correlations('brainsize.txt')
