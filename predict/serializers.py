from rest_framework import serializers
from .models import PlantImage
import base64
from drf_extra_fields.fields import Base64ImageField

class PhotoSerializer(serializers.ModelSerializer):
	class Meta:
		model = PlantImage
		fields = '__all__'

	def create(self, validated_data):
		img=validated_data.pop('img')
		return PlantImage.objects.create(img=image)