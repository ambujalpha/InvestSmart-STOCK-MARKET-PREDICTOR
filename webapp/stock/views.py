from django.shortcuts import render
from .models import Stock
from main.models import Main


def stock(request):

    site = Main.objects.get(pk=2)
    allstocks = Stock.objects.all()
    return render(request, 'front/stock.html', {'site': site, 'allstocks': allstocks})


def stock_detail(request, word):

    site = Main.objects.get(pk=2)
    showstock = Stock.objects.filter(name=word)
    return render(request, 'front/stock_detail.html', {'site': site, 'showstock': showstock})
