import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

##app = Flask(__name__)
# 저장된 모델 불러오기
model = joblib.load('./model/rf_weather_model.joblib')

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        new_data = pd.DataFrame({
            'MEM_NO': [data['memNo']],
            'HOUSE_TYPE': [data['houseType']],
            'AVGTEM': [data['avgTem']],
            'AVGHUM': [data['avgHum']],
            'DT_MM': [data['dtMm']],
            'DAY_WEEK': [data['dayWeek']],
        })
        prediction = model.predict(new_data)
        prediction_str = np.round(prediction, decimals=2).astype(str)  # 숫자를 문자열로 변환
        response = {
            'prediction': prediction_str.tolist()
        }
        return jsonify(response)
    except Exception as e:
        error_response = {
            'error': str(e)
        }
        return jsonify(error_response), 500

if __name__ == '__main__':
    app.run(debug=True,host='192.168.0.22',port=5000)
