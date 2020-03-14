from __future__ import unicode_literals
from django.db import models


class About(models.Model):

    Company_name = models.TextField()
    price = models.IntegerField()

    def __str__(self):
        return self.Company_name

