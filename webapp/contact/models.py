from __future__ import unicode_literals
from django.db import models


class ContactForm(models.Model):

    fname = models.CharField(max_length=30)
    lname = models.CharField(max_length=30)
    email = models.CharField(max_length=50)
    subject = models.TextField()
    date = models.CharField(max_length=12, default="")
    time = models.CharField(max_length=12, default="")
    message = models.TextField()

    def __str__(self):
        return self.fname
