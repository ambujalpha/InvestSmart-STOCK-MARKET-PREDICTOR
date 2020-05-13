from __future__ import unicode_literals
from django.db import models


class Stock(models.Model):

    name = models.CharField(max_length=30)
    txt = models.TextField(default="-")
    invest = models.CharField(max_length=15,default="-")
    divest = models.CharField(max_length=15,default="-")
    picname = models.TextField(default="-")
    picurl = models.TextField(default="-")

    def __str__(self):
        return self.name




