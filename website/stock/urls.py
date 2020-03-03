from django.urls import path
from . import views

app_name = 'stock'

urlpatterns = [

    # /stock/
    path('', views.IndexView.as_view(), name='index'),

]
