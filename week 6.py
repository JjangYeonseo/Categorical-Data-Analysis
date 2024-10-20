#1
import pandas as pd
import statsmodels.api as sm

# 데이터셋 불러오기
url = "http://users.stat.ufl.edu/~aa/intro-cda/data/MBTI.dat"
df = pd.read_table(url)

# 'no'와 'yes' 응답 열 추가
df['no'] = df['n'] - df['drink']
df['yes'] = df['drink']

# 이진 변수(EI, SN, TF, JP) 생성
df['EI'] = df['EI'].map({'e': 0, 'i': 1})  # 외향성 = 0, 내향성 = 1
df['SN'] = df['SN'].map({'s': 0, 'n': 1})  # 감각 = 0, 직관 = 1
df['TF'] = df['TF'].map({'t': 0, 'f': 1})  # 사고 = 0, 감정 = 1
df['JP'] = df['JP'].map({'j': 0, 'p': 1})  # 판단 = 0, 인식 = 1

# 독립 변수와 종속 변수 정의
X = df[['EI', 'SN', 'TF', 'JP']]  # 독립 변수
X = sm.add_constant(X)  # 상수항 추가
y = df['yes'] / (df['yes'] + df['no'])  # 종속 변수(음주 확률)

# 로지스틱 회귀 모델 적합
model = sm.GLM(y, X, family=sm.families.Binomial(), weights=df['yes'] + df['no'])
result = model.fit()

# 모델 요약 출력
print(result.summary())

import numpy as np

# 각 MBTI 유형에 대한 예측 확률 계산
df['predicted_prob'] = result.predict(X)

# 예측 확률 출력
print(df[['EI', 'SN', 'TF', 'JP', 'predicted_prob']])

# ENTP 유형에 대한 예측 확률 계산 (EI=0, SN=1, TF=0, JP=1)
ENTP = np.array([1, 0, 1, 0, 1])  # 상수항 포함
ENTP_prob = result.predict(ENTP)

print(f"ENTP 유형의 예측 확률은 {ENTP_prob[0]}입니다.")

# 모든 MBTI 유형에 대한 예측 확률 컬럼 추가
df['predicted_prob'] = result.predict(X)

# 예측 확률을 포함한 전체 테이블 출력
print(df[['EI', 'SN', 'TF', 'JP', 'predicted_prob']])
