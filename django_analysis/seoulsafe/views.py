'''============================================
소스파일 : seoulsafe/views.py
파일설명 : 데이터 분석 리포트 작성
작성자 : 이지운
버전 : 0.1
생성일자 : 2020-07-01
최종 수정 일자 :2020-07-07
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
from plotly.graph_objs._layout import Layout
from plotly.validators.layout import _plot_bgcolor


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
    
    # 출력용 Dataframe
    df_line = data.iloc[::5, :3]
    
    # LineGraph 그리기 호출
    plot_div_line = crimeLineChart(data, "기간", ["합계발생", "합계검거"], title="연도별 범죄 발생")
    
    # BarGraph
    data_t = data.T
    data_t = data_t.iloc[3::2, :]
    print(data_t)
    print(data_t.index)
    
    bar_fig = px.bar(data_t, x=list(data_t.iloc[:, -1]), y=list(data_t.index), orientation='h', opacity=0.8)
    
    bar_fig.layout = Layout(
        title="2018년 범죄별 발생 건수",
        template="plotly_white",
    )
    plot_div_bar = plot(bar_fig, output_type='div')
    
    # dataframe to html
    df_tohtml = df_line.to_html(classes=["table", "table-sm", "table-striped", "table-hover"], index=False)
    
    return render(request, 'report-crime.html', {'df':df_tohtml, 'plot_div_line':plot_div_line, 'plot_div_bar':plot_div_bar})


# Line Chart 그리기
def crimeLineChart(data, x_value, y_value, title):
    line_fig = px.line(data, x=x_value, y=y_value,
                    hover_name=x_value,
                    title=title, )
    
    return plot(line_fig, output_type='div')

# Bar Chart 그리기
def crimeBarChart():
    pass