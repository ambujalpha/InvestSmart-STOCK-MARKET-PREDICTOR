from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^contact/send/$', views.contact_form, name='contact_form'),
]