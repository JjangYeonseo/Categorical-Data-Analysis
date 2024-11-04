#1
import numpy as np
import pandas as pd
import statsmodels.api as sm

# 데이터 준비: 소득과 행복도별 count 데이터를 입력합니다.
data = pd.DataFrame({
    'income': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'happiness': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'count': [6, 43, 75, 6, 113, 178, 6, 57, 117]
})

# 다항 로짓 모델을 위해 행을 반복하여 데이터 확장
data_expanded = data.loc[data.index.repeat(data['count'])].drop(columns='count')

# 소득과 행복도를 더미 변수로 변환 (다항 로짓 모델에서 필요한 설정)
X = pd.get_dummies(data_expanded['income'], drop_first=True)
y = data_expanded['happiness'] - 1  # 0, 1, 2로 변환

# 다항 로짓 모델 적합
model = sm.MNLogit(y, sm.add_constant(X))
result = model.fit()

# 결과 출력
print(result.summary())

# 매우 행복한 카테고리를 기준으로 상대 확률 계산하기
log_odds = result.params
print("\n각 소득 수준에서 상대적 확률 (로짓 변환 값):\n", log_odds)

# 소득이 평균인 경우 매우 행복할 확률 예측
avg_income_x = np.array([1, 1])  # 상수항과 평균 소득(두 번째 소득 수준에 해당)
prob = result.predict(avg_income_x)[0]
print("\n평균 소득의 경우 매우 행복할 확률:", prob[2])

#2
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm

# 데이터 생성
data = pd.DataFrame({
    'income': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'happiness': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'count': [6, 43, 75, 6, 113, 178, 6, 57, 117]
})

# 데이터 확장: count 열을 기반으로 각 행을 반복
data_expanded = data.loc[data.index.repeat(data['count'])].drop(columns='count')

# 순서형 로짓 모델을 위한 반응 변수와 예측 변수 설정
data_expanded['happiness'] = data_expanded['happiness'].astype('category')
data_expanded['income'] = data_expanded['income'].astype('category')

# 순서형 로짓 모델 적합
from statsmodels.miscmodels.ordinal_model import OrderedModel
model = OrderedModel(data_expanded['happiness'], data_expanded['income'].cat.codes, distr='logit')
result = model.fit(method='bfgs')

# 결과 출력
print(result.summary())

# 귀무가설 검정 - 소득과 결혼 행복도의 독립성 검정
# 우도비 통계량 계산: 자유도 1로 소득에 대한 효과 검정
LRT_stat = result.llf - 3.25  # 잔차 편차 3.25 사용
p_value = sm.stats.chisqprob(LRT_stat, df=1)
print("\n우도비 통계량:", LRT_stat)
print("p-값:", p_value)

# 소득이 평균인 경우 예측 확률 계산
income_level = 2  # 소득이 평균인 경우
pred_prob = result.predict([income_level])[0]
print("\n평균 소득의 경우 행복할 확률:", pred_prob)

#3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

# 데이터 생성
data = pd.DataFrame({
    'gender': ['male', 'male', 'female', 'female'],
    'belief': ['no', 'yes', 'no', 'yes'],
    'count': [50, 200, 60, 250]  # 예시 데이터
})

# 로짓 변환을 위해 'gender'와 'belief'를 카테고리형으로 변환
data['gender'] = data['gender'].astype('category')
data['belief'] = data['belief'].astype('category')

# 독립 모델 적합
model = sm.GLM(data['count'], sm.add_constant(pd.get_dummies(data[['gender', 'belief']], drop_first=True)), 
               family=sm.families.Poisson()).fit()
print(model.summary())

# a) 이탈도 검정: deviance (잔차) 및 자유도 확인
deviance = model.deviance
df_resid = model.df_resid
p_value = 1 - chi2.cdf(deviance, df_resid)
print(f"\n이탈도: {deviance}, 자유도: {df_resid}, p-값: {p_value}")

# b) λ^Yj 구하기
lambda_Y1 = 0.0
lambda_Y2 = model.params['belief_yes']  # 'belief_yes'는 사후 세계를 믿는 경우
print(f"\nλ^Y1 (사후 세계 믿지 않음): {lambda_Y1}")
print(f"λ^Y2 (사후 세계 믿음): {lambda_Y2}")
odds_ratio_belief = np.exp(lambda_Y2)
print(f"사후 세계에 대한 믿음 오즈비: {odds_ratio_belief}")

# c) 포화 모델에서 성별과 사후 세계에 대한 믿음 오즈비
# 포화 모델에서 제공된 성별과 믿음의 상호작용 항의 추정값 사용
lambda_XY_female_yes = 0.1368  # 주어진 성별 여성 및 믿음 "예"의 효과
odds_ratio_gender_belief = np.exp(lambda_XY_female_yes)
print(f"\n성별에 따른 오즈비 (여성의 사후 세계에 대한 믿음 오즈비): {odds_ratio_gender_belief}")

# d) Postlife 데이터셋 독립 모델 적합 (예시 데이터)
# 예시로 데이터 생성, 실제 데이터로 교체 필요
postlife_data = pd.DataFrame({
    'race': ['black', 'black', 'white', 'white', 'other', 'other'],
    'belief': ['no', 'yes', 'no', 'yes', 'no', 'yes'],
    'count': [30, 120, 80, 300, 15, 45]  # 예시 데이터
})

postlife_data['race'] = postlife_data['race'].astype('category')
postlife_data['belief'] = postlife_data['belief'].astype('category')

# 독립 모델 적합
postlife_model = sm.GLM(postlife_data['count'], 
                        sm.add_constant(pd.get_dummies(postlife_data[['race', 'belief']], drop_first=True)), 
                        family=sm.families.Poisson()).fit()
print(postlife_model.summary())

# λ^Y1, λ^Y2 구하기
lambda_Y1_postlife = 0.0
lambda_Y2_postlife = postlife_model.params['belief_yes']
odds_ratio_postlife_belief = np.exp(lambda_Y2_postlife)
print(f"\nPostlife 데이터에서 사후 세계에 대한 믿음 오즈비: {odds_ratio_postlife_belief}")
