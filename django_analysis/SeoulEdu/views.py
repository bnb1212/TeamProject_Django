from django.shortcuts import render
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import pandas as pd
import numpy as np
from tensorflow.python.keras.layers.core import Activation
from dask.dataframe.core import DataFrame
from matplotlib import font_manager, rc
import plotly.express as px

def listFunc(request):
    return render(request, "edulist.html")

def main1Func(request):    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False
    
    df1 = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\SeoulEdu\\files\\AllEdu.xlsx', encoding='utf-8')
    df2 = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\SeoulEdu\\files\\AllMo.xlsx', encoding='utf-8')
    

#     df1 = pd.read_excel("../testdata/AllEdu.xlsx",sheet_name='Sheet1',encoding='utf8') # 자치구별 평생교육 현황
#     df2 = pd.read_excel("../testdata/AllMo.xlsx",sheet_name='Sheet1',encoding='utf8') # 자치구별 총부가가치 자료
    
    #평생교육
    df1 = df1.loc[:,['기간','자치구','기관수','프로그램수','총 수강인원','프1000단']]
    index = df1[(df1['자치구']== '계')].index
    df1 = df1.drop(index)
     
    # print(df1)
     
    grouped = df1['프로그램수'].groupby(df1['자치구'])
    fm = grouped.mean()
    # print(fm)
     
    # print(df1.자치구.unique())
     
    #자치구별 프로그램수 평균
    fm.plot()
    plt.rc('font', family='Malgun gothic')
    a = df1.자치구.unique()
    plt.xticks(range(0,len(df1.자치구.unique())),sorted(a))
    plt.title('자치구별 평균 프로그램수')
    plt.xlabel('자치구')
    plt.ylabel('프로그램수')    
     
    plt.clf()
     
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
     
    grouped2 = df2['지역내요소소득'].groupby(df2['자치구'])
    fm2 = grouped2.mean()
    print(fm2)
     
    #자치구별 요소소득 평균
    fm2.plot()
    plt.rc('font', family='Malgun gothic')
    a2 = df2.자치구.unique()
    plt.xticks(range(0,len(df2.자치구.unique())),sorted(a2))
    plt.title('자치구별 평균 요소소득')
    plt.xlabel('자치구')
    plt.ylabel('요소소득')    

    return render(request, "edumain1.html")

