from django.shortcuts import render
from django.views import generic
from .models import Stks


class IndexView(generic.ListView):
    template_name = 'stock/index.html'

    def get_queryset(self):
        return Stks.objects.all()
