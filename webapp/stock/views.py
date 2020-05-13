from django.shortcuts import render


def stock(request):

    return render(request, 'front/stock.html')


def stock_detail(request, word):

    return render(request, 'front/stock_detail.html')