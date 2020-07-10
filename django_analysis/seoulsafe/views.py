'''============================================
소스파일 : seoulsafe/views.py
파일설명 : 데이터 분석 리포트 작성
작성자 : 이지운
버전 : 1.0
생성일자 : 2020-07-01
최종 수정 일자 :2020-07-010
최종 수정자 : 이지운
최종 수정 내용 : 테스트 완료
============================================'''

from django.shortcuts import render
import pandas as pd
import numpy as np
from django_pandas import io
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from plotly.offline import plot
import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs import Scatter
from plotly.graph_objs import Heatmap
from plotly.graph_objs._layout import Layout
from plotly.validators.layout import _plot_bgcolor
from plotly.graph_objs import Bar

import json
from tensorflow.keras import layers
from django.http.response import HttpResponse

# Create your views here.
def mainFunc(request):
    return render(request, 'safelist.html')


# 범죄율 보고서
def crimeFunc(request):
    # 데이터 불러오기
    # '\\static\\files\\test.csv'
    data = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulsafe\\files\\cri.xlsx', encoding='utf-8')
    # 데이터에 결측값이 있음
    
    # 상관계수
    print(data.corr())
    
    # 데이터 슬라이싱
    data_1 = data.iloc[:, :2]
    data_1 = data_1.set_index("기간")
    # data_1 = data_1.astype(float)
#     print(data_1)
    print(data[data['기간'] == 2000])
    # 출력용 Dataframe
    df_line = data.iloc[::5, :3]
    
    # ------------------ 그래프 ------------------------------------------------
    # 상관계수 Heatmap 그리기
    heat_fig = px.imshow(data.corr(),
                         x=list(data.corr().columns),
                         y=list(data.corr().columns))
    heat_fig.update_layout(title='서울 범죄 발생 통계 히트맵')
    plot_div_heat = plot(heat_fig, output_type='div')
    
    # LineGraph 그리기 호출
    plot_div_line = crimeLineChart(data, "기간", ["합계발생", "합계검거"], title="연도별 범죄 발생")
    
    # BarGraph
    data_t = data[data['기간'] == 2018].T
    data_event = data_t.iloc[3::2, :]
    data_chepo = data_t.iloc[4::2, :]
    
    # 발생건수
    data_event = data_event.iloc[:, -1].to_dict()
    # 검거건수
    data_chepo = data_chepo.iloc[:, -1].to_dict()
    # legend
    data_label = ['강력범', '절도범', '기타형사범', '특별법범', '폭력범', '지능범', '풍속범']
    
    trace_event = go.Bar(name='발생건수', x=list(data_event.values()), y=data_label, orientation='h')
    trace_chepo = go.Bar(name='검거', x=list(data_chepo.values()), y=data_label, orientation='h')
    
    bar_layout = Layout(
        title="2018년 범죄별 발생 건수",
        template="plotly_white",
        showlegend=True,
        barmode='group'
    )
    pdata = [trace_event, trace_chepo]
    bar_fig = go.Figure(data=pdata, layout=bar_layout)
    plot_div_bar = plot(bar_fig, output_type='div')
    
    # dataframe to html
    df_tohtml = df_line.to_html(classes=["table", "table-sm", "table-striped", "table-hover"], index=False)
    
    # --------------------- 예측 --------------------------
    
    return render(request, 'report-crime.html', {'df':df_tohtml, 'plot_div_line':plot_div_line, 'plot_div_bar':plot_div_bar, 'plot_div_heat':plot_div_heat})


# 예측 ajax 통신 
def predCrimeFunc(request):
    data = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulsafe\\files\\cri.xlsx', encoding='utf-8')
    
    new_x, r2_s = predCrime(data, int(request.GET["year"]))
    print(new_x[0][0], r2_s)
    context = {'pred':int(new_x[0][0]), 'r2s':f"{float(r2_s):%}"}
    print(context, type(context))
    print(json.dumps(context), ' ', type(json.dumps(context)))
    return HttpResponse(json.dumps(context), content_type='application/json')


# Line Chart 그리기
def crimeLineChart(data, x_value, y_value, title):
    line_fig = px.line(data, x=x_value, y=y_value,
                    hover_name=x_value,
                    title=title,)
    
    return plot(line_fig, output_type='div')


# 예측 함수
def predCrime(data, input_x):
    
    data_1 = data.iloc[:, :2]
    
    # feature & label 설정
    x_data = data_1.iloc[:, 0]
    print(x_data, type(x_data))
    y_data = data_1.iloc[:, 1]
    print(y_data)

    # train / test 분리
    train_dataset = data_1.sample(frac=0.7, random_state=123)
    test_dataset = data_1.drop(train_dataset.index)
    print(train_dataset.shape)  # (279, 8) -> (274, 8)
    print(test_dataset.shape)  # (119, 8) -> (118, 8)

    # 표준화 준비
    train_stat = train_dataset.describe()
    train_stat.pop('합계발생')
    train_stat = train_stat.transpose()

    # 분리된 dataset에서 label 뽑기
    train_labels = train_dataset.pop('합계발생')
    test_labels = test_dataset.pop('합계발생')

    # 표준화
    st_train_data = st_func(train_dataset, train_stat)
    st_test_data = st_func(test_dataset, train_stat)
    
    model = build_model()

    # 학습 조기 종료 설정
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)  
    
    # 모델 훈련
    epochs = 1000
    history = model.fit(st_train_data, train_labels, epochs=epochs, validation_split=0.2, verbose=2, callbacks=[early_stop])  # 여기에 텐서보드 넣을 수도있음

    # 설명력
    from sklearn.metrics import r2_score
    test_pred = model.predict(st_test_data).flatten()  # 차원 떨어뜨려
    r2_s = r2_score(test_labels, test_pred)
    
    # 새값 예측
    new_x = model.predict(st_func([input_x], train_stat))
    print("새값 : ", new_x)
    
    return new_x, r2_s


# 표준화 처리 함수(요소값 - 평균)/표준편차
def st_func(x, train_stat): 
    return ((x - train_stat['mean']) / train_stat['std'])


# 모델 생성
def build_model():
    network = tf.keras.Sequential([
        layers.Dense(units=128, activation=tf.nn.relu, input_shape=[1]),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation="linear")
    ])
    # 모델 컴파일
    opti = tf.keras.optimizers.Adam(0.01)
    network.compile(optimizer=opti, loss='mse', metrics=['mse', 'mae'])  # mse, mae 평균제곱, 평균절대
    
    return network  # model = network
