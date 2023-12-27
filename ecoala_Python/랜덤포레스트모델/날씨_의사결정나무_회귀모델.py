from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from DBManager import *
from sklearn import datasets
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

# 폰트 설정
##폰트설정
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/H2GTRE.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

mydb = DBManager()

# SQL 쿼리
sql = """
SELECT
dt_dd,
    mem_no,
    house_type,
    round(avg(c.tem), 0) as avgTEM,
    round(avg(HUM), 0) as avgHUM,
    round(avg(c.prec), 0) as avgPREC,
    ROUND(AVG(day_sum_ele), 0) AS day_ELE,
CASE 
        WHEN to_char(to_date(dt_dd, 'yymmdd'), 'd') IN (0, 6) THEN 1
        ELSE 0
    END as DAY_WEEK,    
    to_char(to_date(dt_dd, 'yymmdd'), 'mm') as DT_MM
FROM
    (
        SELECT
            mem_id,
            to_char(use_dt, 'yymmdd') as dt_dd,
            ROUND(sum(dt + aircon + tv + heat + stove + blanket + afry + ahs + other_appliances), 3) as day_sum_ele
        FROM
            mem_app_ele
        GROUP BY
            mem_id, to_char(use_dt, 'yymmdd')
    ) a
    , mem_info b , weather_list c
    WHERE a.mem_id = b.mem_id
    and a.dt_dd = to_char(c.mrd_dt, 'yymmdd')
GROUP BY
    mem_no, house_type, dt_dd
"""

# SQL 결과를 DataFrame으로 읽어오기
df = pd.read_sql(con=mydb.conn, sql=sql)


scaler = MinMaxScaler(feature_range=(0, 1))
scaler_x = MinMaxScaler()
df['DAY_ELE'] = scaler.fit_transform(np.array(df['DAY_ELE'].values.reshape(-1, 1)))

df[['MEM_NO','HOUSE_TYPE','AVGTEM','AVGHUM','DT_MM','DAY_WEEK']]= scaler_x.fit_transform(df[['MEM_NO','HOUSE_TYPE','AVGTEM','AVGHUM','DT_MM','DAY_WEEK']])

x= df[['MEM_NO','HOUSE_TYPE','AVGTEM','AVGHUM','DT_MM','DAY_WEEK']]
y= df[['DAY_ELE']]

# 데이터를 훈련 세트와 나머지로 나눔
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.4)

# 다시 나눈 데이터를 출력
print(f'Training set shape: {x_train.shape}')
print(f'Test set shape: {x_test.shape}')

# 랜덤 포레스트 회귀 모델 생성
model = DecisionTreeRegressor(random_state=42)

# 훈련 세트로 모델 훈련
model.fit(x_train, y_train)

# 훈련 세트로 예측
y_pred_train = model.predict(x_train)
# 테스트 세트로 예측
y_pred_test = model.predict(x_test)

# 훈련 세트의 성능 평가
mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

# 테스트 세트의 성능 평가
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print(f'Train MSE: {mse_train:.2f}')
print(f'Train R-squared: {r2_train:.2f}')

print(f'Test MSE: {mse_test:.2f}')
print(f'Test R-squared: {r2_test:.2f}')

##튜닝
dt_model = DecisionTreeRegressor(random_state=42)

# 탐색할 하이퍼파라미터 설정
param_grid = {
    'n_estimators': [100],
    'max_depth': [30],
    'min_samples_split': [3],
    'min_samples_leaf': [1]
}

# GridSearchCV를 사용하여 최적의 하이퍼파라미터 조합 탐색
scoring = ['neg_mean_squared_error', 'r2', 'explained_variance']
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, scoring=scoring, cv=5, refit='r2')
grid_search.fit(x_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters: ", grid_search.best_params_)

# 최적의 모델을 사용하여 예측
y_pred_train_optimal = grid_search.best_estimator_.predict(x_train)
y_pred_test_optimal = grid_search.best_estimator_.predict(x_test)

# 최적의 모델 평가: 평균 제곱 오차(Mean Squared Error, MSE)
mse_train_optimal = mean_squared_error(y_train, y_pred_train_optimal)
mse_test_optimal = mean_squared_error(y_test, y_pred_test_optimal)

print(f'Optimal Training MSE: {mse_train_optimal:.2f}')
print(f'Optimal Test MSE: {mse_test_optimal:.2f}')

# 최적의 모델 정확도 평가
r2_train_optimal = r2_score(y_train, y_pred_train_optimal)
r2_test_optimal = r2_score(y_test, y_pred_test_optimal)

print(f'Optimal Training R-squared: {r2_train_optimal:.2f}')
print(f'Optimal Test R-squared: {r2_test_optimal:.2f}')

## 모델 저장
joblib.dump(grid_search.best_estimator_, 'dt_weather_model.joblib')