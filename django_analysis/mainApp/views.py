from django.shortcuts import render

# Create your views here.
def indexFunc(request):
    return render(request, "index.html")