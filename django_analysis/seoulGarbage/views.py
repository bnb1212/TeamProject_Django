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
# Create your views here.

def mainFunc(request):
    
    plt.rc('font', family='malgun gothic')
    plt.rc('xtick', labelsize=7)
    #df = pd.read_csv('./files/reportreform.csv',encoding='euc-kr')
    df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\files\\reportreform.csv', encoding = 'euc-kr')
#    with open('reportreform.csv', encoding='euc-kr') as csvfile:
#        rr = csv.reader(csvfile, delimiter=',')
#        print(rr)

    



    df = df.astype({'발생량': np.float, 
                

                    '재활용': np.float,
                    '음식물': np.float,
                    '소각': np.float,
                    '매립': np.float,
                    '정도': np.float,})




    print('------')
    print(df.info())
    '''
    #print(df)#읽어오기 
    print(df[df['구분'] == '용산구'])
    print(df.구분.unique())#['종로구' '중구' '용산구' '성동구' '광진구' '동대문구' '중랑구' '성북구' '강북구' '도봉구' '노원구' '은평구'
    # '서대문구' '마포구' '양천구' '강서구' '구로구' '금천구' '영등포구' '동작구' '관악구' '서초구' '강남구' '송파구'
    # '강동구']
    print('----------------------')
    df1 = df[df['구분'] == '용산구']
    df1 = df1.astype({'발생량': np.float})
    print(df1)
    print(type(df1))
    print()
    print(df1.발생량.sum())
    '''
    #년도별
    #yearx = df.기간.unique()
    #print(yearx)
    #df.astype({'발생량': np.float})
    #print(df[df['기간']==2018]['발생량'].sum())
    #yeary = [11438.3, 11968.19, 12052.3, 12058.30, 11672.69, 11170.2, 11420.0, 11525.0, 11446.90, 11336.8, 10020.4, 9340.1, 9189.30,8558.99,9613.80,9438.69,9608.0,9217.3,9492.9]
    #dfyear = pd.read_csv('yearsum.csv',encoding='euc-kr')
    dfyear = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\files\\yearsum.csv', encoding = 'euc-kr')
    #df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\files\\reportreform.csv', encoding = 'euc-kr')
    print(dfyear)

    sns.barplot(data = dfyear, x = "기간",y="발생량",)
    plt.show()

    sum_by_gu = df.groupby('구분').발생량.mean()
    print(sum_by_gu)
    label = ['종로' ,'중구' ,'용산' ,'성동', '광진' ,'동대문', '중랑' ,'성북', '강북' ,'도봉', '노원' ,'은평',
             '서대문', '마포' ,'양천' ,'강서', '구로' ,'금천' ,'영등포', '동작', '관악', '서초', '강남','송파',
             '강동']

    index = np.arange(len(label))

    print(label)
    plt.figure(figsize=(10,3))

    plt.bar(index, sum_by_gu)

    plt.title('mean of garbage ')

    plt.xlabel('gu')

    plt.ylabel('sum of garbage')

    plt.xticks(index, label)

    plt.show()



    print('datasetinfo--------------')
    #df2 = pd.read_csv('reportreformobject.csv',encoding='euc-kr')
    df2 = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\files\\reportreformobject.csv.csv', encoding = 'euc-kr')
    #df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '\\static\\files\\reportreform.csv', encoding = 'euc-kr')

    dataset=df2.values


    x = dataset[:,2:6]#feature
    print(x)
    y = dataset[:,6]#label
    print(y)



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
    history = model.fit(x, y, validation_split=0.3, epochs=500, batch_size=10)#모델학습하면서 

    loss, acc = model.evaluate(x,y, verbose=0)
    print('훈련된  모델 정확도 : {:5.2f}%'.format(acc * 100))
    new_data = np.array([[100.0, 138.0, 335.0, 120.0]])
    #result = model.predict(new_data)
    #print('예측 결과 (고유치): ', result)
    #print('예측 결과(0 or 1) : ', np.squeeze(np.where(result > 0.5, 1, 0)))
    #print('예상 수치: ', model.predict(x[1:])) 

    #print('예측결과 : ', np.where(pred.flatten() > 0.5, 1, 0))
    #pred = model.predict(np.array([[1,2],[10,5]]))
    #print('예측 결과 : ', pred)     # [[0.16490099] [0.9996613 ]]
    #print('예측 결과 : ', np.squeeze(np.where(pred > 0.5, 1, 0)))  # [0 1]

    new_x = [[23.3, 3.7, 166.3, 143.5]]
    pred = model.predict(new_x)
    print('예측 결과 : ',pred)
    print('예측결과 : ', np.where(pred.flatten() > 0.5, 1, 0))

    new_x1 = [[13.4, 34.7, 47.3, 123.5]]
    pred1 = model.predict(new_x1)
    print('예측 결과 : ',pred1)
    print('예측결과 : ', np.where(pred1.flatten() > 0.5, 1, 0))


    new_x2 = [[273.4, 113.7, 6.3, 1.5]]
    pred2 = model.predict(new_x2)
    print('예측 결과 : ',pred2)
    print('예측결과 : ', np.where(pred2.flatten() > 0.5, 1, 0))

    new_x3 = [[13.4, 280.7, 18.3, 63.5]]
    pred3 = model.predict(new_x3)
    print('예측 결과 : ',pred3)
    print('예측결과 : ', np.where(pred3.flatten() > 0.5, 1, 0))


    '''
    dataset = df.values
    print(dataset)
    print()
    x2000 = dataset[0:25,4:9]#feature->11열까지   -2
    y2000 = dataset[0:25,2]#label(class) ->제일 마지막 열  -1
    print(x2000)
    print(y2000)

    x2001 = dataset[25:50,4:9]#feature->11열까지
    y2001 = dataset[25:50,2]#label(class) ->제일 마지막 열
    print(x2001)
    print(y2001)
    '''
    
    return render(request, "edumain.html")