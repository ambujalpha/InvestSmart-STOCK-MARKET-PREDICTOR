from django.shortcuts import render, get_object_or_404, redirect
from .models import About


def home(request):

    return render(request, 'front/index.html')


def about(request):

    return render(request, 'front/about.html')


def product(request):

    return render(request, 'front/products.html')