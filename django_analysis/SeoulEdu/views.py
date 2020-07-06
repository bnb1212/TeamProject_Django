from django.shortcuts import render

# Create your views here.

def listFunc(request):
    return render(request, "edulist.html")

def main1Func(request):
    joy1()
    return render(request, "edumain1.html")

def joy1():
    print('joyjojojo')
    # 범주형, 연속형자료 어디에 속하는지 알아야함
    # 독립은 하나인데 종속이 여러개