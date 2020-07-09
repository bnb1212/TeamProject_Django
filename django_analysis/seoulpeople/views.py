from django.shortcuts import render
import pandas as pd
from django_pandas import io
import os
from plotly.offline import plot
import plotly.express as px
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objects as go
import plotly.express as px
from django.http.response import HttpResponse

# 데이터 읽어오고 슬라이싱 및 공동 데이터
data = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulpeople\\files\\report.xlsx', encoding='utf-8')
data1 = data[data['자치구'] != '합계'] #합계는 제외
year_sum = data[data['자치구'] == '합계'] # 합계만 가져옴
data_label = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구',
'금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구',
'서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구',
'은평구', '종로구', '중구', '중랑구']

def main_Func(request):
    return render(request, 'peoplelist.html')

def list_Func(request):
    # 데이터 만들기
    data_min = pd.DataFrame()
    df_test = data1.groupby('기간').남자합계.mean() #기간만 빼올려고 만듬
    data_min['기간'] = df_test.index #index만 꺼내옴
    data_min['남자평균'] = data1.groupby('기간').남자합계.mean().values #기간별 남자 평균
    data_min['여자평균'] = data1.groupby('기간').여자합계.mean().values #기간별 여자 평균
    

    # 시각화
    #연도별 남여 평균  
    line_fig = px.line(data_min, x='기간', y=['남자평균', '여자평균'], title='기간별 남녀 평균')
    plot_div = plot(line_fig, output_type='div')
    
    #인구 총합
    bar_fig = px.bar(x=year_sum.기간, y=year_sum.합계)
    bar_fig.layout = Layout(
        title="연도별 서울 총 인구 합(1992-2019)",
        template="plotly_white",
    )
    plot_div_bar = plot(bar_fig, output_type='div')
    
    createModel()
    
    return render(request, 'report-people.html', {'plot_div_bar':plot_div_bar, 'plot_div':plot_div})

def yearFM_Func(request):
    year = request.GET['year']
    year = int(year)
    
    data1 = data[data['자치구'] != '합계']
    data1 = data1[data1['기간'] ==  year] #아작스에서 가져온 값의 기간만 슬라이싱 
    label=data_label
    
    data_Msum = data1.groupby('자치구').남자합계.sum()
    data_Fsum = data1.groupby('자치구').여자합계.sum()
    
    fig = go.Figure(data=[
    go.Bar(name='남자', x=label, y=data_Msum),
    go.Bar(name='여자', x=label, y=data_Fsum)
    ])
    
    fig.update_layout(barmode='group', title=str(year)+"년 남자,여자의 수") #타이틀에 int변수를 strig변수로 변환후 년도 출력 
    plot_div_bar = plot(fig, output_type='div')

    return HttpResponse(plot_div_bar)



def future_Func(request):
    future = request.GET['f'] #아작스 값 가져오기
    gu = request.GET['gu']
    future = int(future) #int로 변환
    
    

def createModel(request):
    '''
    all_data = data1[all_data.자치구]
    
    for len(data_label) in all_data:
        
    '''
    



