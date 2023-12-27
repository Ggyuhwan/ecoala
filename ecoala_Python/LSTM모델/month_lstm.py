from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import pickle
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
# Scaler 객체를 파일로 저장
with open('eco_month_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
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


# 데이터 준비
seq_len = 60
future_period = 1
result = []
# # 각 주소별로 데이터 처리
unique_id = df['MEM_ID'].unique()

for mem_id in unique_id:
    # 주소별 데이터 추출
    mem_data = df[df['MEM_ID'] == mem_id]
    mem_data = mem_data.sort_values(by=['MEM_ID', 'MONDT'])
    data_cnt = len(mem_data['MONELE'])
    for idx in range(data_cnt - seq_len - future_period):
        seq_x = mem_data['MONELE'][idx: idx + seq_len]
        seq_y = mem_data['MONELE'][idx + seq_len: idx + seq_len + future_period]
        result.append(np.append(seq_x, seq_y))

result = np.array(result)
row_cnt = int(round(result.shape[0] * 0.9))

# 훈련 및 테스트 데이터 분할
train_data = result[:row_cnt, :]
x_train = train_data[:, :seq_len]
y_train = train_data[:, seq_len:]
x_train_reshape = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = result[row_cnt:, :seq_len]
x_test_reshape = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
y_test = result[row_cnt:, seq_len:]


# LSTM 모델 정의
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(seq_len, future_period))))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(future_period, activation='linear'))
model.add(Dropout(0.3))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
# 기존의 출력 레이어
model.add(Dense(units=1))
model.compile(loss='mse', optimizer='adam')
# 모델 훈련
model.fit(x_train_reshape, y_train, validation_data=(x_test_reshape, y_test), batch_size=60, epochs=200)

# 모델 저장
model.save('ala200_2.h5')

# 예측 및 시각화
pred = model.predict(x_test_reshape)
#
plt.figure(figsize=(20, 10))
for i in range(future_period):
    plt.subplot(future_period, 1, i + 1)
    plt.plot(y_test[:, i], label='True')
    plt.plot(pred[:, i], label='Prediction')
    plt.title(f"Day {i + 1}")
    plt.legend()
plt.tight_layout()
plt.show()