from django.shortcuts import render
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping
import seaborn as sns
import csv
from plotly.offline import plot
import plotly.express as px
from plotly.graph_objs import Scatter
import json
from django.shortcuts import render

from django.http.response import HttpResponse
# Create your views here.

def mainFunc(request):
    
    plt.rc('font', family='malgun gothic')#한글 폰트를 위한 선언
    plt.rc('xtick', labelsize=7)#그래프가 깨지지 않게 하기 위한 사이즈 조절 

    df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulGarbage\\files\\reportreform.csv', encoding = 'euc-kr')
    # csv file reading

    df = df.astype({'발생량': np.float, 
                    '재활용': np.float,
                    '음식물': np.float,
                    '소각': np.float,
                    '매립': np.float,
                    '정도': np.float,})#통계를 내기 위한 데이터 타입 변환

    #print('------')
    #print(df.info())#변환이 되었는지 확인하기.
   
   
    dfyear = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulGarbage\\files\\yearsum.csv', encoding = 'euc-kr')
    #년도별 쓰레기 통계를 위한 데이터 csv file read
    #print(dfyear)
    

    
    # 시각화 (bar chart)
    x_data = dfyear["기간"]
    y_data = dfyear["발생량"]
    plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='red')],
               output_type='div')#lines 그래프를 그려줌

    sum_by_gu = df.groupby('구분').발생량.mean()#구별로 쓰레기양의 평균치를 구해줌
    #print(sum_by_gu)
    label = ['종로' ,'중구' ,'용산' ,'성동', '광진' ,'동대문', '중랑' ,'성북', '강북' ,'도봉', '노원' ,'은평',
             '서대문', '마포' ,'양천' ,'강서', '구로' ,'금천' ,'영등포', '동작', '관악', '서초', '강남','송파',
             '강동']#구별 라벨

    index = np.arange(len(label))

    #print(label)
  
      # 시각화 (bar chart)
    x_data2 = label
    y_data2 = sum_by_gu
    plot_div2 = plot([Scatter(x=x_data2, y=y_data2,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='blue')],
               output_type='div')#구별 쓰레기양 평균을 그래프 (lines)으로 만들어줌


    #print('datasetinfo--------------')
    #df2 = pd.read_csv('reportreformobject.csv',encoding='euc-kr')
    
    df2 = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulGarbage\\files\\reportreformobject.csv', encoding = 'euc-kr')
    #df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\files\\reportreform.csv', encoding = 'euc-kr')

    dataset=df2.values

    x = dataset[:,2:6]#feature
    print(x)
    y = dataset[:,6]#label
    print(y)
    
    model = Sequential()
    model.add(Dense(32, input_dim=4, activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))#2개중 하나만 나오늘 출력일때는 sigmoid 여러개중 하나 나오면 softmax



    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x, y, validation_split=0.3, epochs=30, batch_size=10)#모델학습하면서 
    

    loss, acc = model.evaluate(x,y, verbose=0)
    print('훈련된  모델 정확도 : {:5.2f}%'.format(acc * 100))
    model.save('mnist_model.hdf5')
    del model

    
    #return render(request, "edumain.html")
    return render(request, 'garbagemain.html', {'plot_div2':plot_div2, 'plot_div':plot_div})
    #return render(request, 'report-crime.html', {'df':df_tohtml, 'plot_div':plot_div})
    
def Func1(request):
    #'recycle':recycle,'food':food, 'burn':burn, 'land':land
    recycle = float(request.GET['recycle'])
    food = float(request.GET['food'])
    burn = float(request.GET['burn'])
    land = float(request.GET['land'])
    #print(msg, type(msg))
    #print(recycle, type(recycle))
    #print(food, type(food))
    #print(burn, type(burn))
    #print(land, type(land))
    df2 = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulGarbage\\files\\reportreformobject.csv', encoding = 'euc-kr')
    
    dataset=df2.values


    x = dataset[:,2:6]#feature
    print(x)
    y = dataset[:,6]#label
    print(y)

    '''

    
    #모델
    model = Sequential()
    model.add(Dense(32, input_dim=4, activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))#2개중 하나만 나오늘 출력일때는 sigmoid 여러개중 하나 나오면 softmax



    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x, y, validation_split=0.3, epochs=50, batch_size=10)#모델학습하면서 
    '''
    
    model = tf.keras.models.load_model('mnist_model.hdf5')#위에서 학습한 모델을 불러온다 -> ajax 써도 끊기지 않음

    #loss, acc = model.evaluate(x,y, verbose=0)
    #print('훈련된  모델 정확도 : {:5.2f}%'.format(acc * 100))
 

    new_x = [[recycle, food, burn, land]]
    pred = model.predict(new_x)
    print('예측 결과 : ',pred)
    print('예측결과 : ', np.where(pred.flatten() > 0.5, 1, 0))
    predict_int = np.where(pred.flatten() > 0.5, 1, 0)

    predict_str = ""
    if (predict_int >= 1):
        predict_str = "쓰레기 양이 많습니다. 환경을 위해 쓰레기를 줄여주세요.!!"
    else:
        predict_str = "예상 오염도가 적습니다. 앞으로도 환경에 신경써주세요. 감사합니다!"
    
    
    loss, acc = model.evaluate(x,y, verbose=0)
    acc = round(acc * 100)
    context = {'key':predict_str, 'acc':acc}
    print(context,type(context))
    print(json.dumps(context), ' ', type(json.dumps(context)))
    return HttpResponse(json.dumps(context), content_type='application/json')
    
    