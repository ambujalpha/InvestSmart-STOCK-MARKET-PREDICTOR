from django.conf.urls import url
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [

    url(r'^stock/$', views.stock, name='stock'),
    url(r'^stock/(?P<word>.*)/$', views.stock_detail, name='stock_detail'),

]
