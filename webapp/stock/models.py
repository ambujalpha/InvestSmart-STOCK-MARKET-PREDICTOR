from __future__ import unicode_literals
from django.db import models


class Stock(models.Model):

    name = models.CharField(max_length=30)
    txt = models.TextField(default="-")

    def __str__(self):
        return self.name




