from django.shortcuts import render
from main.models import Main


def home(request):

    site = Main.objects.get(pk=2)
    return render(request, 'front/home.html', {'site': site})

