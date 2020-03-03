from django.db import models


class Stks(models.Model):
    price = models.IntegerField()
    company_name = models.CharField(max_length=100)
    current_price = models.IntegerField()
    original_price = models.IntegerField()

    objects = models.Manager()

    def __str__(self):
        return self.price


