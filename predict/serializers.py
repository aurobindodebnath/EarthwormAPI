from rest_framework import serializers
from .models import PlantImage, Post
import base64
from drf_extra_fields.fields import Base64ImageField

class PhotoSerializer(serializers.ModelSerializer):
	class Meta:
		model = PlantImage
		fields = '__all__'

	def create(self, validated_data):
		img=validated_data.pop('img')
		return PlantImage.objects.create(img=image)

class PostSerializer(serializers.ModelSerializer):
	img = serializers.SerializerMethodField('get_image_url')
	class Meta:
		model = Post
		fields = ('title','description','img','publish')

	def get_image_url(self, obj):
		return obj.image.url