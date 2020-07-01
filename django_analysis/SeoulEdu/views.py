from django.shortcuts import render

# Create your views here.

def listFunc(request):
    return render(request, "edulist.html")

def main1Func(request):
    joy1()
    return render(request, "edumain1.html")

def joy1():
    print('joyjojojo')