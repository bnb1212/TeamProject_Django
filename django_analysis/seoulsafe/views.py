from django.shortcuts import render
import pandas as pd
import os
# Create your views here.
def mainFunc(request):
    return render(request, 'safelist.html')

def crimeFunc(request):
    # '\\static\\files\\test.csv'
    data = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\files\\cri.xlsx', encoding='utf-8')
    print(data)
    return render(request, 'report-crime.html')
    