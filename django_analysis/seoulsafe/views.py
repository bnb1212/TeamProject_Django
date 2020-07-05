from django.shortcuts import render
import pandas as pd
from django_pandas import io
import os
# Create your views here.
def mainFunc(request):
    return render(request, 'safelist.html')

def crimeFunc(request):
    # '\\static\\files\\test.csv'
    data = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + '\\static\\seoulsafe\\files\\cri.xlsx', encoding='utf-8')
    print(data)
    df_tohtml = data.to_html(classes="table table-sm")
    return render(request, 'report-crime.html', {'df':df_tohtml})
    