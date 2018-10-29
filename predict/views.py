from django.shortcuts import render
import uuid
import tensorflow as tf
import numpy as np
import os,glob,cv2
import numpy as np
from .models import PlantImage
from rest_framework.decorators import api_view
import base64
import random, string
from rest_framework.response import Response
from .serializers import PhotoSerializer
from django.views.decorators.csrf import csrf_exempt
from sem7 import settings
from .diseaseIndex import disease_index, alldis
#disease_index = ("Healthy Cherry ","Apple with Black Rot","Apple with Scab","Cherry with Powdery Mildew","Tomato with Mosaic Virus","Tomato with Leaf Mold","Healthy Potato","Healthy Grape","tomato with Spider Mites","Corn with Gray Leaf Spot","Tomato with Septoria Leaf Spot","Grape with Leaf Blight","Healthy Strawberry","Strawberry with Leaf Scorch","Potato with Late Blight","Healthy Corn","Healthy Apple","Pepper with Bacterial Spot","Healthy Blueberry","Corn with Northern Leaf Blight","Grape Black Measels","Apple Cedar Rust","Potato Early Blight","Tomato with Target Spots","Corn with Common Rust","Healthy Raspberry","Tomato with Early Blight","Grape with Black Rot","Healthy Pepper","Healthy Tomato")

graph = tf.get_default_graph()
sess = tf.Session()
saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Test_Model.meta")
saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/"))
y_pred = graph.get_tensor_by_name("y_pred:0")
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 39)) 

def id_generator(size=7, chars=string.ascii_uppercase + string.digits):
	return str(uuid.uuid4())

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadView(request):
	if request.method == 'GET':
		images = PlantImage.objects.all()
		serializer = PhotoSerializer(images, many=True)
		return Response(serializer.data)
	if request.method=='POST':
		print(settings.BASE_DIR)
		data = request.data
		image=base64.b64decode(data["img"])
		name = id_generator()
		f= open(settings.BASE_DIR + "/media/"+name+".jpg", "wb")
		f.write(image)
		f.close()
		obj = PlantImage.objects.create(img=name+".jpg")
		obj.save()
		filename = settings.BASE_DIR + "/media/"+name+".jpg"
		image_size=128
		num_channels=3
		images = []
		image = cv2.imread(filename)
		image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
		images.append(image)
		images = np.array(images, dtype=np.uint8)
		images = images.astype('float32')
		images = np.multiply(images, 1.0/255.0) 
		x_batch = images.reshape(1, image_size,image_size,num_channels)
		global graph
		with graph.as_default():
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index[max_index]
				obj.accuracy = str(round(max_value*100,2))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})

@api_view(['GET', ])
@csrf_exempt
def diseaseInfo(request):
	return Response(alldis)



'''
	dir_path = os.path.dirname(os.path.realpath(__file__))
	print(dir_path)
	filename = dir_path +'/' +"a.jpg"
	image_size=128
	num_channels=3
	images = []
	image = cv2.imread(filename)
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	images.append(image)
	images = np.array(images, dtype=np.uint8)
	images = images.astype('float32')
	images = np.multiply(images, 1.0/255.0) 
	x_batch = images.reshape(1, image_size,image_size,num_channels)
	global graph
	with graph.as_default():
		sess = tf.Session()
		saver = tf.train.import_meta_graph(dir_path+'/model.meta')
		saver.restore(sess, tf.train.latest_checkpoint(dir_path+'/'))
		y_pred = graph.get_tensor_by_name("y_pred:0")
		x= graph.get_tensor_by_name("x:0") 
		y_true = graph.get_tensor_by_name("y_true:0") 
		y_test_images = np.zeros((1, 6))#len(os.listdir('training_data')))) 
		feed_dict_testing = {x: x_batch, y_true: y_test_images}
		result=sess.run(y_pred, feed_dict=feed_dict_testing)
		print(result)
		return HttpResponse("Hi")
'''