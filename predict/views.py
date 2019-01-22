from django.shortcuts import render
import uuid
import math
import tensorflow as tf
import numpy as np
import os,glob,cv2
import numpy as np
from .models import PlantImage, Post
from rest_framework.decorators import api_view
import base64
import random, string
from rest_framework.response import Response
from .serializers import PhotoSerializer, PostSerializer
from django.views.decorators.csrf import csrf_exempt
from sem7 import settings
from .diseaseIndex import disease_index_grape,disease_index_apple,disease_index_strawberry,disease_index_cherry,disease_index_corn,disease_index_potato,disease_index_peach,disease_index_pepper,disease_index_tomato,disease_index_other, alldis
#disease_index = ("Healthy Cherry ","Apple with Black Rot","Apple with Scab","Cherry with Powdery Mildew","Tomato with Mosaic Virus","Tomato with Leaf Mold","Healthy Potato","Healthy Grape","tomato with Spider Mites","Corn with Gray Leaf Spot","Tomato with Septoria Leaf Spot","Grape with Leaf Blight","Healthy Strawberry","Strawberry with Leaf Scorch","Potato with Late Blight","Healthy Corn","Healthy Apple","Pepper with Bacterial Spot","Healthy Blueberry","Corn with Northern Leaf Blight","Grape Black Measels","Apple Cedar Rust","Potato Early Blight","Tomato with Target Spots","Corn with Common Rust","Healthy Raspberry","Tomato with Early Blight","Grape with Black Rot","Healthy Pepper","Healthy Tomato")

graph_apple = tf.Graph()
graph_grape = tf.Graph()
graph_strawberry = tf.Graph()
graph_cherry = tf.Graph()
graph_corn = tf.Graph()
graph_potato = tf.Graph()
graph_tomato = tf.Graph()
graph_other = tf.Graph()
graph_peach = tf.Graph()
graph_pepper = tf.Graph()


def id_generator(size=7, chars=string.ascii_uppercase + string.digits):
	return str(uuid.uuid4())


#grape

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewGrape(request):
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
		global graph_grape 
		with graph_grape.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/grape/grape_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/grape/"))
			y_pred = graph_grape.get_tensor_by_name("y_pred:0")
			x= graph_grape.get_tensor_by_name("x:0") 
			y_true = graph_grape.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 5))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_grape[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#apple

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewApple(request):
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
		image_size=256
		num_channels=3
		images = []
		image = cv2.imread(filename)
		image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
		images.append(image)
		images = np.array(images, dtype=np.uint8)
		images = images.astype('float32')
		images = np.multiply(images, 1.0/255.0) 
		x_batch = images.reshape(1, image_size,image_size,num_channels)
		global graph_apple 
		with graph_apple.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/apple_files/model_apple.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/apple_files/"))
			y_pred = graph_apple.get_tensor_by_name("y_pred:0")
			x= graph_apple.get_tensor_by_name("x:0") 
			y_true = graph_apple.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 5))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_apple[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#strawberry


@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewStrawberry(request):
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
		global graph_strawberry 
		with graph_strawberry.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Strawberry/strawberry_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Strawberry/"))
			y_pred = graph_strawberry.get_tensor_by_name("y_pred:0")
			x= graph_strawberry.get_tensor_by_name("x:0") 
			y_true = graph_strawberry.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 3))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_strawberry[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#cherry

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadCherry(request):
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
		global graph_cherry 
		with graph_cherry.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Cherry/cherry_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Cherry/"))
			y_pred = graph_cherry.get_tensor_by_name("y_pred:0")
			x= graph_cherry.get_tensor_by_name("x:0") 
			y_true = graph_cherry.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 3))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_cherry[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})



#corn


@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewCorn(request):
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
		global graph_corn 
		with graph_corn.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Corn/corn_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Corn/"))
			y_pred = graph_corn.get_tensor_by_name("y_pred:0")
			x= graph_corn.get_tensor_by_name("x:0") 
			y_true = graph_corn.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 5))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_corn[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#potato


@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewPotato(request):
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
		global graph_potato 
		with graph_potato.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Potato/potato_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Potato/"))
			y_pred = graph_potato.get_tensor_by_name("y_pred:0")
			x= graph_potato.get_tensor_by_name("x:0") 
			y_true = graph_potato.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 4))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_potato[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#tomato

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewTomato(request):
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
		global graph_tomato 
		with graph_tomato.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Tomato/tomato_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Tomato/"))
			y_pred = graph_tomato.get_tensor_by_name("y_pred:0")
			x= graph_tomato.get_tensor_by_name("x:0") 
			y_true = graph_tomato.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 5))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_other[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#other

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewOther(request):
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
		global graph_other 
		with graph_other.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Other/other_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Other/"))
			y_pred = graph_other.get_tensor_by_name("y_pred:0")
			x= graph_other.get_tensor_by_name("x:0") 
			y_true = graph_other.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 5))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_other[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#peach

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewPeach(request):
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
		global graph_peach 
		with graph_peach.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Peach/peach_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Peach/"))
			y_pred = graph_peach.get_tensor_by_name("y_pred:0")
			x= graph_peach.get_tensor_by_name("x:0") 
			y_true = graph_peach.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 3))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_peach[max_index]
				obj.accuracy = str(math.floor(max_value*100))
				obj.save()
				return Response({"result": obj.disease+" with accuracy of "+obj.accuracy + " %"})
			else:
				obj.disease = "Unsure of any plant or disease"
				obj.accuracy = "0"
				obj.save()
				return Response({"result": obj.disease})


#pepper

@api_view(['GET', 'POST', ])
@csrf_exempt
def uploadViewPepper(request):
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
		global graph_pepper 
		with graph_pepper.as_default():
			sess = tf.Session()
			saver = tf.train.import_meta_graph(settings.BASE_DIR+"/predict/Pepper/pepper_model.meta",clear_devices=True)
			saver.restore(sess, tf.train.latest_checkpoint(settings.BASE_DIR+"/predict/Pepper/"))
			y_pred = graph_pepper.get_tensor_by_name("y_pred:0")
			x= graph_pepper.get_tensor_by_name("x:0") 
			y_true = graph_pepper.get_tensor_by_name("y_true:0") 
			y_test_images = np.zeros((1, 3))
			feed_dict_testing = {x: x_batch, y_true: y_test_images}
			result=sess.run(y_pred, feed_dict=feed_dict_testing)
			print(result)
			max_value = np.amax(result)
			max_index = np.argmax(result)
			if max_value>0.50:
				obj.disease = disease_index_pepper[max_index]
				obj.accuracy = str(math.floor(max_value*100))
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

@api_view(['GET', ])
@csrf_exempt
def showNews(request):
	posts = Post.objects.all()
	serializer = PostSerializer(posts, many=True)
	return Response(serializer.data)




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