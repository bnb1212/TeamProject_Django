from django.shortcuts import render
import tensorflow as tf
import os
from plotly.offline import plot
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import pandas as pd
import numpy as np
from tensorflow.python.keras.layers.core import Activation
from dask.dataframe.core import DataFrame
from matplotlib import font_manager, rc
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.layers import Input 
from tensorflow.keras.models import Model
import json
from django.http.response import HttpResponse

df1 = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\SeoulEdu\\files\\AllEdu.xlsx', encoding='utf-8') # 자치구별 평생교육 현황
df2 = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\SeoulEdu\\files\\AllMo.xlsx', encoding='utf-8') # 자치구별 총부가가치 자료

#평생교육
df1 = df1.loc[:,['기간','자치구','기관수','프로그램수','총 수강인원','프1000단']]
index = df1[(df1['자치구']== '계')].index
df1 = df1.drop(index)
 
# 자치구별로 프로그램수의 평균을 구한다 
grouped = df1['프로그램수'].groupby(df1['자치구'])
fm = grouped.mean()
a = df1.자치구.unique() 
 
# # 총부가가치 
df2['지역내총부가가치'] = df2['지역내총부가가치'].apply(pd.to_numeric,errors='coerce')
df2 = df2[(df2['경제활동별']== '소계')]
index = df2[(df2['자치구']== '서울시')].index
df2 = df2.drop(index)
index = df2[(df2['기간']== 2011)].index
df2 = df2.drop(index)
index = df2[(df2['기간']== 2017)].index
df2 = df2.drop(index)
index = df2[(df2['기간']== 2018)].index
df2 = df2.drop(index)

# df1,df2 합치기
dff = pd.merge(df1,df2, how='left', left_on=['기간','자치구'], right_on = ['기간','자치구'])
dff = dff.drop(['경제활동별'], axis='columns')
dff = dff[['기관수','프로그램수','총 수강인원','지역내요소소득','지역내총부가가치']]
dff.columns = ['기관수','프로그램수','총수강인원','요소소득','총부가가치']
dff = dff.dropna()

dff['프로그램수'] /= 1000
dff['요소소득'] /= 10000000

def listFunc(request):
    return render(request, "edulist.html")

def main1Func(request):    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 자치구별로 요소소득의 평균을 구한다 
    grouped2 = df2['지역내요소소득'].groupby(df2['자치구'])
    fm2 = grouped2.mean()
    a2 = df2.자치구.unique()
    
    # 차트 설정
    colors=px.colors.sequential.Plasma*3 #색깔 세트 1
    colors2=px.colors.sequential.Viridis*4 #색깔 세트 2
    
    # 프로그램 평균 그려주기
    fig = go.Figure(data=[  
    go.Bar( x=sorted(a), y=fm, marker_color=colors2),
    ])
#     fig.update_layout(title="자치구별 평균 프로그램 수") 
    plot_div_bar2 = plot(fig, output_type='div')
    
    # 요소소득 평균 그려주기
    fig2 = go.Figure(data=[
    go.Bar( x=sorted(a2), y=fm2,marker_color=colors),
    ])
#     fig2.update_layout(title="자치구별 평균 요소소득")
    plot_div_bar = plot(fig2, output_type='div')
    
    # 상관계수 Heatmap 그리기
    heat_fig = px.imshow(dff.corr(),
                         x=list(dff.corr().columns),
                         y=list(dff.corr().columns))
#     heat_fig.update_layout(title='총부가가치와 평생교육기관의 히트맵')
    plot_div_heat = plot(heat_fig,output_type='div')
    
    return render(request, "edumain1.html",{'plot_div_bar_mo':plot_div_bar,'plot_div_bar_pro':plot_div_bar2,'plot_div_heat':plot_div_heat})

# 모델 생성
def pred_model():
    inputs = Input(shape=(1,))
    output1 = Dense(64,activation='linear')(inputs)
    output1 = Dense(128,activation='linear')(output1)
    output1 = Dense(128,activation='linear')(output1)
    output1 = Dense(128,activation='linear')(output1)
    outputs = Dense(1,activation='linear')(output1)
                   
    model = Model(inputs,outputs) 
                   
    opti = optimizers.SGD(lr=0.001) # 이하 방법1과 동일
    model.compile(opti,loss='mse',metrics='mse')

    return model

def predFunc(request):
    
    money = int(request.GET['money'])
     
    x_data = np.array(dff.요소소득,dtype = np.int32) 
    y_data = np.array(dff.프로그램수, dtype=np.int32)
    
    # 모델 불러서 학습시킴
    model = pred_model()            
    model.fit(x = x_data, y=y_data,batch_size = 1, epochs = 150, verbose=3)
    
    # 설명력과 요청된 값의 예측값
    r2_s = r2_score(y_data,model.predict(x_data))
    pred_result = int(1000*model.predict(np.array([money/10000000],dtype = np.int32)))
    
    set = {'pred_result':pred_result, 'r2_s':int(r2_s*100)}
    return HttpResponse(json.dumps(set), content_type='application/json')