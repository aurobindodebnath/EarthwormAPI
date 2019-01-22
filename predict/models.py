from django.db import models

# Create your models here.
class PlantImage(models.Model):
    img = models.ImageField(blank=True)
    disease = models.CharField(max_length=150, null = True)
    accuracy = models.FloatField(null=True, blank=True)
    def __str__(self):
        return str(self.id)

class Post(models.Model):
	title = models.CharField(max_length=120)
	description= models.TextField(max_length=150)
	image = models.ImageField(blank=True, null=True)
	publish = models.DateField(auto_now=False, auto_now_add=False)