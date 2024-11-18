#1
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import logit

# 데이터 입력
data = {
    'Gender': ['Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Male'],
    'Location': ['Rural', 'Rural', 'Urban', 'Urban', 'Rural', 'Rural', 'Urban', 'Urban'],
    'SeatBelt': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Injured': [973, 757, 996, 759, 1084, 513, 812, 380],
    'NotInjured': [3246, 6134, 7287, 11587, 6123, 6693, 10381, 10969]
}

df = pd.DataFrame(data)

# 1단계: S (SeatBelt) ~ G (Gender) + L (Location)
df['Total'] = df['Injured'] + df['NotInjured']
df['SeatBeltBinary'] = (df['SeatBelt'] == 'Yes').astype(int)

logit_model_1 = logit("SeatBeltBinary ~ Gender + Location", data=df)
result_1 = logit_model_1.fit()
print(result_1.summary())

# 2단계: I (Injured) ~ G (Gender) + L (Location) + S (SeatBelt)
df['InjuryRate'] = df['Injured'] / df['Total']
logit_model_2 = logit("InjuryRate ~ Gender + Location + SeatBeltBinary", data=df)
result_2 = logit_model_2.fit()
print(result_2.summary())

#2
import numpy as np
import pandas as pd

# 데이터 정의 (예: Table 7.17)
data = np.array([
    [50, 32, 25, 18],  # Row 1
    [40, 35, 30, 20],  # Row 2
    [30, 28, 25, 15],  # Row 3
    [20, 18, 12, 10]   # Row 4
])

# 데이터프레임 생성
df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3", "Col4"])
print(df)

from scipy.stats import chi2_contingency

# 카이제곱 검정
chi2, p, dof, expected = chi2_contingency(data)

# 결과 출력
print(f"G² 값 (Deviance): {chi2}")
print(f"p-value: {p}")
print(f"자유도: {dof}")
print(f"기대빈도:\n{expected}")

import statsmodels.api as sm

# 행과 열 점수 정의
row_scores = np.arange(1, data.shape[0] + 1)  # 행 점수: [1, 2, 3, 4]
col_scores = np.arange(1, data.shape[1] + 1)  # 열 점수: [1, 2, 3, 4]

# 독립형 모델 GLM 적합
model = sm.GLM(data.ravel(), 
               sm.add_constant(np.outer(row_scores, col_scores).ravel()), 
               family=sm.families.Poisson()).fit()

# 결과 출력
print(model.summary())

# 로그 오즈 비율 계산 함수
def local_log_odds_ratio(table, row1, col1, row2, col2):
    # (셀[row1, col1] * 셀[row2, col2]) / (셀[row1, col2] * 셀[row2, col1])
    odds_ratio = (table[row1, col1] * table[row2, col2]) / (table[row1, col2] * table[row2, col1])
    log_odds_ratio = np.log(odds_ratio)
    return log_odds_ratio

# 예시: 열 1-2와 열 2-3 간의 로컬 로그 오즈 비율
log_odds_1_2 = local_log_odds_ratio(data, 0, 0, 0, 1)
log_odds_2_3 = local_log_odds_ratio(data, 0, 1, 0, 2)

print(f"열 1-2의 로그 오즈 비율: {log_odds_1_2}")
print(f"열 2-3의 로그 오즈 비율: {log_odds_2_3}")

# 새로운 열 점수 정의
col_scores_new = np.array([1, 2, 4, 5])

# GLM 모델 재적합
model_new = sm.GLM(data.ravel(), 
                   sm.add_constant(np.outer(row_scores, col_scores_new).ravel()), 
                   family=sm.families.Poisson()).fit()

# 결과 출력
print(model_new.summary())

