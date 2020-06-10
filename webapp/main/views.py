from django.shortcuts import render
from main.models import Main


def home(request):

    site = Main.objects.get(pk=2)
    return render(request, 'front/home.html', {'site': site})


def consult(request):

    return render(request, 'front/consult.html')


def market(request):

    return render(request, 'front/market.html')


def monitoring(request):

    return render(request, 'front/monitoring.html')


def investment(request):

    return render(request, 'front/investment.html')


def management(request):

    return render(request, 'front/management.html')