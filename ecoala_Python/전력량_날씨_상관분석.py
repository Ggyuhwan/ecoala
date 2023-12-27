import datetime
import os

import pandas as pd
from matplotlib import pyplot as plt
from DBManager import *
from sklearn import datasets
import seaborn as sns
from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/H2GTRE.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

mydb = DBManager()
sql = """
select MONTHPRCE,MONTHHUM,MONTHTEM,round(avg(MONTHEle),3) as MONTHELE
from(
        SELECT TO_CHAR(TRUNC(mrd_dt), 'yymmdd') AS MONTHDate,
               ROUND(AVG(prec), 0) AS MONTHPRCE,
               ROUND(AVG(hum), 0) AS MONTHHUM,
               ROUND(AVG(tem), 0) AS MONTHTEM
        FROM weather_list a, mem_info b
        WHERE b.mem_id in ('2499535076','2510000133','2110020035','2398710232','2498535024','2499820333',
            '2510000139','2398200152','2397510132','2398590084','2397580147','2297300005',
            '2398180102','2595910008','2410030341','2397270153','2510040123','2498010005',
            '2498010034','2510040079','2210020019','2397205059','2398710236','2410020393',
            '2398210223','2397310011')
         AND a.region = b.region
        AND TRUNC(to_char(MRD_DT,'YYMM'))> TRUNC(to_char(SYSDATE,'YYMM'))-3
        GROUP BY TRUNC(mrd_dt)
        ORDER BY TRUNC(mrd_dt) DESC) z,
        (
        SELECT TO_CHAR(USE_DT,'yymmdd') as MONTHDate
                ,ROUND(sum(dt + aircon + tv + heat + stove + blanket + afry + ahs + other_appliances), 3) as MONTHEle
        FROM MEM_APP_ELE
        WHERE MEM_ID in ('2499535076')
        AND TRUNC(to_char(USE_DT,'YYMM'))> TRUNC(to_char(SYSDATE,'YYMM'))-1
        GROUP BY TO_CHAR(USE_DT,'yymmdd')
        ORDER BY 1 ) x
        where z.MONTHDate = x.MONTHDate
        group by MONTHPRCE,MONTHHUM,MONTHTEM
"""

df = pd.read_sql(con=mydb.conn, sql=sql)
# 데이터 타입 변환
#df["MONTH_1"] = pd.to_datetime(df["MONTH_1"], format='%y-%m')

df_corr = df.corr()

print(df_corr)
cols=['MONTHPRCE','MONTHHUM', 'MONTHTEM', 'MONTHELE']
#온도
#cols=['WEEKTEM', 'WEEKELE']


print(df[cols].head())
sns.pairplot(df[cols])
plt.show()

plt.plot(df)
plt.show()