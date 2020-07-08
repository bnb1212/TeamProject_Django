from django.shortcuts import render
import pandas as pd
from django_pandas import io
import os
from plotly.offline import plot
import plotly.express as px
from plotly.graph_objs import Scatter, Layout
import plotly.graph_objects as go
import plotly.express as px

# Create your views here.


def main_Func(request):
    return render(request, 'peoplelist.html')


def list_Func(request):
    # 파일 읽어오기
    data = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulpeople\\files\\report.xlsx', encoding='utf-8')
    # print(data[:5])
    
    # 데이터 슬라이싱
    data1 = data[data['자치구'] != '합계']
    # print(data1[:5])
    data_sum = pd.DataFrame(data1.groupby('자치구').합계.sum())
    data_Mmin = data1.groupby('자치구').남자합계.mean()
    data_Fmin = data1.groupby('자치구').여자합계.mean()
    
    
    year_sum = data[data['자치구'] == '합계']
    print(year_sum)
    
    #print(data_Mmin)
    #print(data_Fmin)
    data_label = ['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구',
'금천구', '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구',
'서초구', '성동구', '성북구', '송파구', '양천구', '영등포구', '용산구',
'은평구', '종로구', '중구', '중랑구']
    #print(data_label)
    #print(data_sum)
    
    # 시각화
    '''
    bar_fig1 = px.bar(data1, x=data_label, y=[data_Mmin,data_Fmin])
    
    bar_fig1.layout = Layout(
        title="여자 남자 평균",
        template="plotly_white",
        legend('남자','여자'),
        #showlegend=False,
    )
    plot_div_bar1 = plot(bar_fig1, output_type='div')
    '''
    
    animals=data_label

    fig = go.Figure(data=[
    go.Bar(name='남자', x=animals, y=data_Mmin),
    go.Bar(name='여자', x=animals, y=data_Fmin)
    ])

    fig.update_layout(barmode='group', title="~(년도)년  남자,여자의 수")
    plot_div_bar1 = plot(fig, output_type='div')
    
    # BarGraph
    
    #data_t = data.T
    #data_t = data_t.iloc[3::2, :]
    #print(data_t)
    #print(data_t.index)
    
    bar_fig = px.bar(x=year_sum.기간, y=year_sum.합계)
    
    bar_fig.layout = Layout(
        title="연도별 서울 총 인구 합(1992-2019)",
        template="plotly_white",
    )
    plot_div_bar = plot(bar_fig, output_type='div')
    
    '''
    plot_div = px.line(data1, x="자치구", y=["남자합계","여자합계"],
                    title="지역별 남자 여자 합계")
    plot_div_line = plot(plot_div, output_type='div')
    '''
    df_tohtml = data_sum.to_html(classes=["table", "table-sm", "table-striped", "table-hover"])
    
    # return render(request, 'report-crime.html', {'df':df_tohtml, 'plot_div':plot_div})
    return render(request, 'report-people.html', {'df':df_tohtml,'plot_div_bar':plot_div_bar,'plot_div_bar1':plot_div_bar1})
