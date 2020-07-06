'''============================================
소스파일 : seoulsafe/views.py
파일설명 : 데이터 분석 리포트 작성
작성자 : 이지운
버전 : 0.1
생성일자 : 2020-07-01
최종 수정 일자 :2020-07-06
최종 수정자 : 이지운
최종 수정 내용 : 시각화 테스트
============================================'''

from django.shortcuts import render
import pandas as pd
from django_pandas import io
import os
from plotly.offline import plot
import plotly.express as px
from plotly.graph_objs import Scatter


# Create your views here.
def mainFunc(request):
    return render(request, 'safelist.html')


# 범죄율 보고서
def crimeFunc(request):
    # 데이터 불러오기
    # '\\static\\files\\test.csv'
    data = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulsafe\\files\\cri.xlsx', encoding='utf-8')
    print(data)
    
    # 데이터 슬라이싱
    data_1 = data.iloc[:, :2]
    data_1 = data_1.set_index("기간")
    # data_1 = data_1.astype(float)
    print(data_1)
    
    # 시각화 (bar chart)
    x_data = [0, 1, 2, 3]
    y_data = [x ** 2 for x in x_data]
    plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='green')],
               output_type='div')
    
    df_tohtml = data_1.to_html(classes=["table", "table-sm", "table-striped", "table-hover"])
    return render(request, 'report-crime.html', {'df':df_tohtml, 'plot_div':plot_div})
    
