from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from DBManager import *
# 데이터베이스 연결 및 쿼리 수행
mydb = DBManager()

# SQL 쿼리
sql = """
SELECT mem_id, to_char(use_dt,'yymmdd') as MONDT,
           ROUND(sum(dt + aircon + tv + heat + stove + blanket + afry + ahs + other_appliances), 2) AS MONELE
    FROM mem_app_ele a
    where mem_id in ('2499535076','2510000133','2110020035','2398710232','2498535024','2499820333',
    '2510000139','2398200152','2397510132','2398590084','2397580147','2297300005',
    '2398180102','2595910008','2410030341','2397270153','2510040123','2498010005',
    '2498010034','2510040079','2210020019','2397205059','2398710236','2410020393',
    '2398210223','2397310011')
    AND to_char(use_dt,'mm') = to_char(sysdate,'mm')
    GROUP BY mem_id,to_char(use_dt,'yymmdd')
    ORDER BY mem_id, mondt
"""

# SQL 결과를 DataFrame으로 읽어오기
df = pd.read_sql(con=mydb.conn, sql=sql)
columns = ['MEM_ID', 'MONDT', 'MONELE']

# LSTM 모델을 위한 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
df['MONELE'] = scaler.fit_transform(np.array(df['MONELE'].values.reshape(-1, 1)))

# 현재 날짜 설정
current_date = datetime.now().strftime('%y%m%d')
if datetime.now().month == 12:
    last_day_of_month = (datetime(datetime.now().year + 1, 1, 1) - timedelta(days=1)).strftime('%y%m%d')
else:
    last_day_of_month = (datetime(datetime.now().year, datetime.now().month + 1, 1) - timedelta(days=1)).strftime('%y%m%d')


# 남은 월의 날짜 범위 설정 (현재 날짜부터 해당 월의 마지막 날짜까지)
remaining_days = (datetime.strptime(last_day_of_month, '%y%m%d') - datetime.now()).days
prediction_dates = [datetime.strptime(current_date, '%y%m%d') + timedelta(days=i) for i in range(1, remaining_days + 1)]
prediction_dates = [date.strftime('%y%m%d') for date in prediction_dates]



# 데이터를 train과 test로 나누기
train_size = int(len(df['MONELE']) * 0.9)
train, test = df['MONELE'][:train_size], df['MONELE'][train_size-1:]

# LSTM 모델 학습 데이터 생성
data_cnt = len(train)
result_train = []
for idx in range(data_cnt):
    result_train.append(train[idx])

result_train = np.array(result_train)
row_cnt_train = int(round(result_train.shape[0] * 0.9))  # 90% 학습 데이터로 사용

train_data = result_train[:row_cnt_train]
x_train = train_data[:-1]
y_train = train_data[1:]

# LSTM 모델 정의 및 학습
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train.reshape(-1, 1, 1), y_train, epochs=100, batch_size=1)

# 모델 저장
model.save('ecoLSTM.model')

# 테스트 데이터에 대한 예측 수행
result_test = []
for idx in range(row_cnt_train, len(test)):
    result_test.append(test[idx])

result_test = np.array(result_test)

# 데이터 준비
test_input = np.array(test[:-1])
test_input = np.reshape(test_input, (len(test_input), 1, 1))

# 예측 수행
prediction_result_test = model.predict(test_input)

# 스케일 역변환
prediction_result_test = scaler.inverse_transform(prediction_result_test)

# 결과 시각화
fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(np.arange(len(train)), train, label='Training Data')
ax.plot(np.arange(len(train), len(train) + len(test)), test, label='Test Data', linestyle='dashed')
ax.plot(np.arange(len(train), len(train) + len(test)), prediction_result_test, label='Prediction on Test Data', linestyle='dashed')
ax.legend()
plt.show()

# R2 스코어 계산
r2 = r2_score(result_test, prediction_result_test)
print(f'R-squared (R2) Score on Test Data: {r2}')

