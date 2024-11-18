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
import pandas as pd
import numpy as np

# 데이터 프레임 생성
data = {
    "R": np.repeat(range(1, 10), 4),  # 종교 출석도 (1~9 반복)
    "T": list(range(1, 5)) * 9,       # 청소년 피임 태도 (1~4 반복)
    "count": [49, 49, 19, 9, 31, 27, 11, 11, 46, 55, 25, 8, 34, 37, 19, 7, 
              21, 22, 14, 16, 26, 36, 16, 16, 8, 16, 15, 11, 32, 65, 57, 61, 
              4, 17, 16, 20],  # 관측 빈도
}

df = pd.DataFrame(data)  # 데이터 프레임 생성

import statsmodels.api as sm
import statsmodels.formula.api as smf

# 독립성 모델 (factor를 통해 범주형 처리)
model_ind = smf.glm(formula="count ~ C(R) + C(T)", data=df, family=sm.families.Poisson()).fit()
print(model_ind.summary())  # 모델 요약 출력

# R과 T를 실수형으로 변환
df['R'] = df['R'].astype(float)
df['T'] = df['T'].astype(float)

# 선형-선형 연관성 모델
model_ass = smf.glm(formula="count ~ C(R) + C(T) + R:T", data=df, family=sm.families.Poisson()).fit()
print(model_ass.summary())  # 모델 요약 출력

# 독립성 모델과 선형-선형 연관성 모델의 편차 비교
reduction_in_deviance = model_ind.deviance - model_ass.deviance
df_diff = model_ind.df_resid - model_ass.df_resid
p_value = 1 - sm.stats.chisqprob(reduction_in_deviance, df_diff)

print(f"편차 감소: {reduction_in_deviance}, p-value: {p_value}")

