from django.shortcuts import render, get_object_or_404, redirect
from .models import Stock


def home(request):

    site = Stock.objects.get(pk=2)
    return render(request, 'front/index.html', {'site': site})


def about(request):

    site = Stock.objects.get(pk=2)
    return render(request, 'front/about.html', {'site': site})


def product(request):

    site = Stock.objects.get(pk=2)
    return render(request, 'front/products.html', {'site': site})