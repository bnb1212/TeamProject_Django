from django.shortcuts import render

# Create your views here.
def indexFunc(request):
    return render(request, "index.html")

def joeyPageFunc(request):
    return render(request, "properties-single.html")