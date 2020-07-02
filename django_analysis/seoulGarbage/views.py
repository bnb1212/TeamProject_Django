from django.shortcuts import render

# Create your views here.

def mainFunc(request):
    return render(request, 'garbage_main.html')