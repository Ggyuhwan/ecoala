import pandas as pd
import numpy as np
import pickle
from DBManager import *
from keras.models import load_model
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app)  # CORS 설정 추가

sql = {"오늘제외":"""
                SELECT to_char(use_dt,'yymmdd') as ele_day
                     , ROUND(sum(dt + aircon + tv + heat + stove + blanket + afry + ahs + other_appliances), 2) AS MONELE
                FROM mem_app_ele
                WHERE mem_id = :1
                AND use_dt >= SYSDATE - 60
                AND use_dt < TRUNC(SYSDATE)
                GROUP BY to_char(use_dt,'yymmdd')
                ORDER BY 1 
                """
        ,"대상일자":"""
                    SELECT TO_CHAR(dt, 'yymmdd') AS ele_day
                    FROM (
                        SELECT TRUNC(SYSDATE) + LEVEL - 1 AS dt
                        FROM dual
                        CONNECT BY LEVEL <= LAST_DAY(SYSDATE) - TRUNC(SYSDATE) + 1
                    )
                    ORDER BY ele_day
        """
       }
mydb = DBManager()
# 저장된 스케일러 객체 로드
with open('eco_month_scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

model = load_model('./ala200_2.h5')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def main():
    data = request.json
    print(data['memId'])
    df = pd.read_sql(con=mydb.conn, sql=sql['오늘제외'], params=[data['memId']])
    df_day = pd.read_sql(con=mydb.conn, sql=sql['대상일자'])

    print(df.head())
    df['MONELE'] = loaded_scaler.transform(df['MONELE'].values.reshape(-1, 1))
    result = np.array(df['MONELE'].tolist())
    result_reshape = result.reshape((1, 60, 1))

    # 예측 및 시각화
    arr = []
    for i, v in df_day.iterrows():
        print(v['ELE_DAY'])
        if i == 0:
            pred = model.predict([result_reshape])
            arr.append(pred[0, 0])
        else:
            a_rolled = result_reshape[0, 1:, :]
            a_updated = np.concatenate((a_rolled, np.array(arr[-1]).reshape(1, 1)), axis=0)
            a_final = a_updated.reshape((1, 60, 1))
            pred = model.predict([a_final])
            arr.append(pred[0, 0])

    # pred_array = np.array([p[0] for p in arr]).reshape(-1, 1)
    # Now apply the inverse_transform method of your loaded scaler
    inversed_pred = loaded_scaler.inverse_transform(np.array(arr).reshape(1, -1))
    print(inversed_pred)
    # Convert DataFrame to JSON
    pred_result = {'PRED_ELE': inversed_pred.flatten().tolist()
                   ,'ELE_DAY': df_day['ELE_DAY'].tolist() }
    json_result = json.dumps(pred_result)
    return json_result

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")





