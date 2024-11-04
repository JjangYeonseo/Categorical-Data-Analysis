#b
# 필요한 라이브러리 로드
library(MASS)

# 데이터 불러오기 및 확인
data <- read.table("C:/Users/jys20/Desktop/studentsdata.txt", header = TRUE, sep = "")
head(data)  # 데이터가 올바르게 로드되었는지 확인

# 필요시 특정 열 제외 (예: subject 열이 존재하는 경우)
data <- subset(data, select = -c(subject))

# 모든 설명 변수를 포함한 로지스틱 회귀 모델 적합
full_model <- glm(abor ~ ., data = data, family = binomial)

# stepAIC를 사용하여 최적 모델 선택 (AIC 최소화 기준)
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)

# 선택된 모델 결과 요약 출력
summary(step_model)

# 선택된 모델의 AIC 확인
cat("최종 선택된 모델의 AIC:", step_model$aic, "\n")

#c
# 라이브러리 로드
library(lmtest)

# 귀무 모델 적합 (veg에 대한 설명 변수가 없는 모델)
null_model <- glm(veg ~ 1, data = data, family = binomial)

# 전체 모델 적합 (veg에 대한 모든 설명 변수 포함)
full_model <- glm(veg ~ ., data = data, family = binomial)

# 가능도 비율 검정 수행
lr_test <- lrtest(null_model, full_model)

# 검정 결과 전체 보기
print(lr_test)

# p-value 출력
cat("Likelihood-Ratio Test p-value:", lr_test$p.value, "\n")
