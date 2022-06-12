from django.db import models

class Class1(models.Model):
    name = models.CharField(max_length=200)
    language = models.CharField(max_length=100)
    price = models.CharField(max_length=20)

    def __str__(self):
        return self.name
# Create your models here.
