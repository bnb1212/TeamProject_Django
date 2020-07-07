from django.shortcuts import render

def listFunc(request):
    return render(request, "edulist.html")

def main1Func(request):
    joy1()
    return render(request, "edumain1.html")

