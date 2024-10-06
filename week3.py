#1
import pandas as pd
import numpy as np
import statsmodels.api as sm

# 제공된 데이터를 바탕으로 데이터셋 구성
data = {
    'SnoringLevel': [0, 2, 4, 5],     # 코골이 정도 (0, 2, 4, 5)
    'HeartDisease_Yes': [34, 35, 21, 30],  # 심장병 있는 환자 수
    'HeartDisease_No': [1355, 603, 192, 224] # 심장병 없는 환자 수
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 심장병 발생 확률 계산
df['Total'] = df['HeartDisease_Yes'] + df['HeartDisease_No']
df['HeartDisease_Prob'] = df['HeartDisease_Yes'] / df['Total']

# 데이터 확장 (로지스틱 회귀에 맞게 Yes/No 데이터 생성)
expanded_data = pd.DataFrame({
    'SnoringLevel': np.repeat(df['SnoringLevel'].values, df['Total'].values),
    'HeartDisease': np.repeat([1, 1, 1, 1], df['HeartDisease_Yes'].values).tolist() +
                    np.repeat([0, 0, 0, 0], df['HeartDisease_No'].values).tolist()
})

# 절편 추가
expanded_data['intercept'] = 1

# 로지스틱 회귀 모델 적합
logit_model = sm.Logit(expanded_data['HeartDisease'], expanded_data[['intercept', 'SnoringLevel']])
result = logit_model.fit()

# 모델 요약 출력
print(result.summary())

# SnoringLevel의 계수 추출 및 오즈비 계산
beta_snoring = result.params['SnoringLevel']
odds_ratio = np.exp(beta_snoring)
print(f"코골이 정도 1단위 증가에 따른 오즈비: {odds_ratio:.4f}")


#2

import pandas as pd
import statsmodels.api as sm

# Data: Temperatures and whether thermal distress occurred (TD)
data = {'Temp': [66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58],
        'TD': [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]}

df = pd.DataFrame(data)

# Add constant for the intercept
df['intercept'] = 1

# Fit the logistic regression model
logit_model = sm.Logit(df['TD'], df[['intercept', 'Temp']])
result = logit_model.fit()

# Print summary of the model
print(result.summary())


# Coefficients from the fitted model
beta_0 = result.params['intercept']
beta_1 = result.params['Temp']

# Temperature at Challenger flight
temp_challenger = 31

# Calculate the probability of thermal distress at 31°F
prob_challenger = 1 / (1 + np.exp(-(beta_0 + beta_1 * temp_challenger)))
print(f"Probability of thermal distress at 31°F: {prob_challenger:.4f}")


# Find the temperature where the probability is 0.50
temp_50_prob = -beta_0 / beta_1
print(f"Temperature where probability equals 0.50: {temp_50_prob:.2f}°F")


#3
from scipy.stats import chi2

# Deviance 값 설정
null_deviance = 8.3499
residual_deviance = 1.3835

# Deviance 차이 (카이제곱 통계량)
chi_squared_stat = null_deviance - residual_deviance

# 자유도 설정
df = 2

# P-값 계산
p_value = chi2.sf(chi_squared_stat, df)

print(f"카이제곱 통계량: {chi_squared_stat:.4f}")
print(f"P-값: {p_value:.4f}")

import pandas as pd

# 데이터셋 설정
data = {
    'yes': [14, 32, 8, 12],   # AIDS 증상이 나타난 환자 수
    'no': [93, 81, 52, 43],   # 증상이 나타나지 않은 환자 수
    'azt': [1, 0, 1, 0],      # AZT 치료 여부 (1 = 즉시 치료, 0 = 지연 치료)
    'racewhite': [1, 1, 0, 0] # 인종 (1 = 백인, 0 = 흑인)
}

df = pd.DataFrame(data)

# 총 환자 수
df['total'] = df['yes'] + df['no']
df['response'] = df['yes'] / df['total']

# 로지스틱 회귀 모델 설정
import statsmodels.api as sm

# 절편 추가
df['intercept'] = 1

# 로지스틱 회귀 모델 적합
logit_model = sm.Logit(df['yes'] / df['total'], df[['intercept', 'azt', 'racewhite']], weights=df['total'])
result = logit_model.fit()

# 요약 결과 출력
print(result.summary())
