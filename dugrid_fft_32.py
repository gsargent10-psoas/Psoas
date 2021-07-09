from __future__ import print_function, division
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['CUDA_VISIBLE_DEVICES']='0'
import multiprocessing as mp
import scipy
import scipy.misc
import scipy.io as sio
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#from keras.datasets import mnist
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import os
from os import system
from glob import glob
#from loadmicrogriddata import MicrogridDataLoader
from helperfunctions import HelperFunctions
from dataset import Dataset
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image
from keras import backend as k
from tensorflow.python.client import device_lib
from keras.layers.core import Lambda
from keras.models import load_model
from numpy import load
from numpy import vstack
from numpy.random import randint
import math as m
import shutil

class MicrogridGAN() :
	def __init__(self,mode=1,dataFolder='/app/data/',saveFolder='/app/results/',modelFolder='/app/results/',model='model_300.h5',epochs=300,comments='') :
		self.img_rows=256
		self.img_cols=256
		self.pad=0
		self.input_channels=6
		self.output_channels=6
		self.input_img_shape=(self.img_rows,self.img_cols,self.input_channels)
		self.output_img_shape=(self.img_rows,self.img_cols,self.output_channels)
		self.minibatchSize=2
		self.mode=mode
		self.dataFolder=dataFolder
		self.saveFolder=saveFolder
		self.modelFolder=modelFolder
		self.epochs=epochs
		self.comments=comments
		self.sample_interval=10
		self.model = model
		# Number of filters in the first layer of the generator and discriminator
		self.gf=64 #64
		self.df=64 #64

		if (self.mode==1) :
			print("----------Training Settings--------------")
			print("[Data Folder]",self.dataFolder)
			print("[Save Folder]",self.saveFolder)
			print("[Model Folder]",self.modelFolder)
			print("[Rows x Cols]",self.img_rows,"x",self.img_cols)
			print("[Minibatch Size]",self.minibatchSize)
			print("[Input Channels]",self.input_channels)
			print("[Output Channels]",self.output_channels)

			# Calculate output shape of the discriminator (PatchGAN)
			patch = int(16)
			self.disc_patch=(patch,patch,1)
	

			# Optimizer
			optimizer = Adam(0.0002,0.5)
			
			# Build and compile the discrimnator
			print('-----------Discriminator Layers-------------')
			self.discriminator=self.build_modified_patchgan_discriminator()
			self.discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])

			# Build the generator
			print('-----------Generator Layers-------------')
			self.generator=self.build_modified_pix2pix_generator()
			#self.generator=load_model('%s%s'%(self.saveFolder,self.model),custom_objects={'tf': tf, 'm':m})

			img_output=Input(shape=self.output_img_shape)
			img_input=Input(shape=self.input_img_shape)

			fake_output=self.generator(img_input)

			self.discriminator.trainable=False

			valid=self.discriminator([fake_output,img_input])

			self.combined=Model(inputs=[img_output,img_input],outputs=[valid,fake_output])

			self.combined.compile(loss=['mse',HelperFunctions.mae_loss_fft],loss_weights=[1,100],optimizer=optimizer,run_eagerly=True)
			input('Press any key to continue...')
			self.train()
		
		if (self.mode==2) :
			print("----------Testing Settings--------------")
			print("[Data Folder]",self.dataFolder)
			print("[Save Folder]",self.saveFolder)
			print("[Model Folder]",self.modelFolder)
			print("[Rows x Cols]",self.img_rows,"x",self.img_cols)
			print("[Minibatch Size]",1)
			print("[Input Channels]",self.input_channels)
			print("[Output Channels]",self.output_channels)
			patch = int(4)
			self.disc_patch = (patch,patch,1)
			optimizer=Adam(0.0002,0.5)
			self.discriminator = self.build_modified_patchgan_discriminator()
			self.discriminator.compile(loss='mse',optimizer=optimizer,metrics=['accuracy'])
			self.generator=load_model('%s%s'%(self.saveFolder,self.model),custom_objects={'tf': tf, 'm':m})
			img_output = Input(shape=self.output_img_shape)
			img_input = Input(shape=self.input_img_shape)
			gen_img = self.generator(img_input)
			self.discriminator.trainable=False
			valid=self.discriminator([gen_img,img_input])
			self.combined = Model(inputs=[img_output,img_input],outputs=[valid,gen_img])
			self.combined.compile(loss=['mse','mae'],loss_weights=[1,100],optimizer=optimizer)
			input("Press any key to continue...")
			self.test()

	def build_modified_patchgan_discriminator(self) :
		def d_layer(layer_input,filters,fsize=4,strides=1,bn=True,name='name me') :
			d=Conv2D(filters,kernel_size=fsize,strides=strides,padding='same',name=name)(layer_input)
			d=LeakyReLU(alpha=0.2)(d)
			if bn:
				d=BatchNormalization(momentum=0.8)(d)
			return d
		img_output=Input(shape=self.output_img_shape)
		img_input=Input(shape=self.input_img_shape)

		# Concatenate image and conditioning image by channels to produce input
		combined_imgs=Concatenate(axis=-1)([img_output,img_input])
		print(combined_imgs.shape)
		d1=d_layer(combined_imgs,self.df,fsize=4,strides=2,bn=False,name="dl1")
		print(d1.shape)
		d2=d_layer(d1,self.df*2,fsize=4,strides=2,bn=True,name="dl2")
		print(d2.shape)
		d3=d_layer(d2,self.df*4,fsize=4,strides=2,bn=True,name="dl3")
		print(d3.shape)
		d4=d_layer(d3,self.df,fsize=4,strides=2,bn=True,name="dl4")
		print(d4.shape)
		validity=Conv2D(1,kernel_size=2,strides=1,padding='same',activation='sigmoid',name='dlfinal')(d4)
		print(validity.shape)

		return Model([img_output,img_input],validity)

	def build_modified_pix2pix_generator(self) :
		def conv2d(layer_input,filters,fsize=3,bn=True,name='name me') :
			layer_input=Lambda(HelperFunctions.pad3)(layer_input)
			d = Conv2D(filters,kernel_size=fsize,strides=2,padding='valid',name=name)(layer_input)
			d = LeakyReLU(alpha=0.2)(d)
			if bn:
				d=BatchNormalization(momentum=0.8)(d)
			return d

		def deconv2d(layer_input,skip_input,filters,fsize=3,dropout_rate=0,name='name me') : 
			u=UpSampling2D(size=2)(layer_input)
			u=Lambda(HelperFunctions.pad3)(u)
			u=Conv2D(filters,kernel_size=fsize,strides=1,padding='valid',activation='relu',name=name)(u)
			if dropout_rate:
				u=Dropout(dropout_rate)(u)
			u=BatchNormalization(momentum=0.8)(u)
			u=Concatenate()([u,skip_input])
			return u
		
		d0=Input(shape=self.input_img_shape)
		print(d0.shape)

		d1=conv2d(d0,self.gf,bn=False,name='gl1')
		print(d1.shape)
		d2=conv2d(d1,self.gf*2,bn=True,name='gl2')
		print(d2.shape)
		d3=conv2d(d2,self.gf*4,bn=True,name='gl3')
		print(d3.shape)
		d4=conv2d(d3,self.gf*8,bn=True,name='gl4')
		print(d4.shape)
		#d5=conv2d(d4,self.gf*8,bn=True,name='gl5')
		#print(d5.shape)

		#u1=deconv2d(d5,d4,self.gf*8,name='gl6')
		#print(u1.shape)
		#u2=deconv2d(u1,d3,self.gf*4,name='gl7')
		u2=deconv2d(d4,d3,self.gf*4,name='gl7')
		print(u2.shape)
		#u3=deconv2d(u2,d2,self.gf*2,name='gl8')
		u3=deconv2d(u2,d2,self.gf*2,name='gl8')
		print(u3.shape)
		u4=deconv2d(u3,d1,self.gf,name='gl9')
		print(u4.shape)

		u5=UpSampling2D(size=2)(u4)
		u5=Lambda(HelperFunctions.pad3)(u5)
		print(u5.shape)
		output_img=Conv2D(self.output_channels,kernel_size=3,strides=1,padding='valid',activation='tanh',name='gout')(u5)
		print(output_img.shape)
		
		return Model(d0,output_img)

	def imread(self,path) :
		return np.array(list(sio.loadmat(path,verify_compressed_data_integrity=False).values())[3]).astype(np.float64)

	def imgscale(self,img,stype="minmax",param1=0,param2=1) :
		np.seterr(divide='ignore',invalid='ignore')
		[M,N] = img.shape
		img = img.astype(np.float64)
		if(stype=="minmax") :
			alpha_l = img[:].min()
			alpha_h = img[:].max()
		elif(stype=="statistical") :
			mu = img[:].mean()
			sd = img[:].std()
			alpha_l = mu - param1*sd
			alpha_h = mu + param2*sd
		elif(stype=="absolute") :
			alpha_l = param1
			alpha_h = param2

		# Saturate all values of in < alpha_l = alpha_l
		img[img[:]<alpha_l] = alpha_l
		# Saturate all values of in > alpha_h = alpha_h
		img[img[:]>alpha_h] = alpha_h
		# Linearly scale data in the range of [alpha_l,alpha_h] --> [0,1]
		out = np.nan_to_num((img[:] - alpha_l)/(alpha_h - alpha_l))
		out = np.reshape(out,(M,N))
		return out 

	def load_training_data_fft(self) :
		ugrid = sorted(glob('%strain/I*.mat' % (self.dataFolder)))
		stokes = sorted(glob('%strain/S*.mat' % (self.dataFolder)))
		print('%strain/S*.mat' % (self.dataFolder))
		n_batches = int(len(ugrid))
		for u,s in zip(ugrid,stokes):
			start_pix_x = randint(1,2448-256)
			start_pix_y = randint(1,2048-256)
			ugrid_name = os.path.basename(u)
			print(ugrid_name)
			stokes_name = os.path.basename(s)
			print(stokes_name)
			input_img = tf.zeros([1,self.img_rows,self.img_cols,self.input_channels],dtype=tf.float64)
			gt_img = tf.zeros([1,self.img_rows,self.img_cols,self.output_channels],dtype=tf.float64)
			for i in range(self.minibatchSize):
				while start_pix_x % 3 != 0 or start_pix_y % 2 != 0:
					start_pix_x = randint(1,2448-256)
					start_pix_y = randint(1,2048-256)
				ugrid_img = tf.cast(np.pad(self.imread(u),[[0,0],[0,self.pad]],'symmetric')[start_pix_y:start_pix_y+self.img_rows,start_pix_x:start_pix_x+self.img_cols],tf.complex64)
				stokes_img = tf.cast(np.pad(self.imread(s),[[0,0],[0,self.pad],[0,0]],'symmetric')[start_pix_y:start_pix_y+self.img_rows,start_pix_x:start_pix_x+self.img_cols,:],tf.complex64)
				normalization = tf.cast(tf.reduce_sum(tf.abs(ugrid_img)),tf.complex64)

				ugrid_fft_img = tf.expand_dims(tf.signal.fft2d(ugrid_img)/normalization,axis=-1)
				s0_fft_img = tf.expand_dims(tf.signal.fft2d(stokes_img[:,:,0])/normalization,axis=-1)
				s1_fft_img = tf.expand_dims(tf.signal.fft2d(stokes_img[:,:,1])/normalization,axis=-1)
				s2_fft_img = tf.expand_dims(tf.signal.fft2d(stokes_img[:,:,2])/normalization,axis=-1)

				ugrid_fft_img_real = tf.cast(tf.math.real(ugrid_fft_img),tf.float64)			
				ugrid_fft_img_imag = tf.cast(tf.math.imag(ugrid_fft_img),tf.float64)
				ugrid_fft_img_cat = tf.concat([ugrid_fft_img_real,ugrid_fft_img_imag],2)
				ugrid_fft_img_cat = tf.concat([ugrid_fft_img_cat,ugrid_fft_img_cat,ugrid_fft_img_cat],2)
				input_img = tf.concat([input_img,tf.expand_dims(ugrid_fft_img_cat,axis=0)],0)
			
				s0_fft_img_real = tf.cast(tf.math.real(s0_fft_img),tf.float64)			
				s0_fft_img_imag = tf.cast(tf.math.imag(s0_fft_img),tf.float64)
				s0_img = tf.concat([s0_fft_img_real,s0_fft_img_imag],2)
				s1_fft_img_real = tf.cast(tf.math.real(s1_fft_img),tf.float64)			
				s1_fft_img_imag = tf.cast(tf.math.imag(s1_fft_img),tf.float64)
				s1_img = tf.concat([s1_fft_img_real,s1_fft_img_imag],2)
				s2_fft_img_real = tf.cast(tf.math.real(s2_fft_img),tf.float64)			
				s2_fft_img_imag = tf.cast(tf.math.imag(s2_fft_img),tf.float64)
				s2_img = tf.concat([s2_fft_img_real,s2_fft_img_imag],2)
				gt_img = tf.concat([gt_img,tf.expand_dims(tf.concat([s0_img,s1_img,s2_img],2),axis=0)],0)
				#gt_img = tf.concat([gt_img,tf.expand_dims(s0_img,axis=0)],0)
			
			print('input_img',input_img[1:,:,:,:].shape)
			print('gt_img',gt_img[1:,:,:,:].shape)
			
			yield input_img[1:,:,:,:].numpy(),gt_img[1:,:,:,:].numpy(),ugrid_name,stokes_name

	def load_test_data(self) :
		ugrid = sorted(glob('%stest/I*.mat' % (self.dataFolder)))
		stokes = sorted(glob('%stest/S*.mat' % (self.dataFolder)))
		print('%stest/S*.mat' % (self.dataFolder))
		n_batches = int(len(ugrid))
		for u,s in zip(ugrid,stokes):
			ugrid_name = os.path.basename(u)
			print(ugrid_name)
			stokes_name = os.path.basename(s)
			print(stokes_name)
			ugrid_img = tf.cast(np.pad(self.imread(u),[[0,0],[0,self.pad]],'symmetric'),tf.complex64)
			stokes_img = tf.cast(np.pad(self.imread(s),[[0,0],[0,self.pad],[0,0]],'symmetric'),tf.complex64)
			yield ugrid_img.numpy(),stokes_img.numpy(),ugrid_name,stokes_name
			
	def train(self):
		start_time = datetime.datetime.now()

		valid=np.ones((self.minibatchSize,)+self.disc_patch)
		fake=np.zeros((self.minibatchSize,)+self.disc_patch)
		disc_acc=np.zeros([self.epochs])
		disc_loss=np.zeros([self.epochs])
		gen_loss=np.zeros([self.epochs])
		d_loss = 0.0
		elapsed_time=datetime.datetime.now()-start_time
		prev_elapsed_time=elapsed_time
		system('clear')
		print('Training...')
		print('Time: %s' %(elapsed_time))
		plt.ion()
		self.fgt = plt.figure()
		self.fs0_gt = plt.imshow(np.random.rand(self.img_rows*self.img_cols).reshape((self.img_rows,self.img_cols)),cmap='gray')
		plt.title('GT S0 FFT')
		self.fgt.canvas.flush_events()
		self.fgt2 = plt.figure()
		self.fs0_gt2 = plt.imshow(np.random.rand(self.img_rows*self.img_cols).reshape((self.img_rows,self.img_cols)),cmap='gray')
		plt.title('GT S0')
		self.fgt2.canvas.flush_events()
		self.fgen = plt.figure()
		self.fs0_gen = plt.imshow(np.random.rand(self.img_rows*self.img_cols).reshape((self.img_rows,self.img_cols)),cmap='gray')
		plt.title('Gen S0 FFT')
		self.fgen.canvas.flush_events()
		self.fgen2 = plt.figure()
		self.fs0_gen2 = plt.imshow(np.random.rand(self.img_rows*self.img_cols).reshape((self.img_rows,self.img_cols)),cmap='gray')
		plt.title('Gen S0')
		self.fgen2.canvas.flush_events()
		gen_img=np.zeros((self.minibatchSize,self.img_rows,self.img_cols,self.output_channels))
		for epoch in range(epochs):
			g_avg = 0
			d_avg = 0
			count = 0
			for batch_i, (input_img,gt_img,ugrid_name,stokes_name) in enumerate(self.load_training_data_fft()):
				gen_img = self.generator.predict(input_img)
				d_loss_real = self.discriminator.train_on_batch([gt_img,input_img],valid)
				d_loss_fake = self.discriminator.train_on_batch([gen_img,input_img],fake)
				d_loss=0.5*np.add(d_loss_real,d_loss_fake)
				
				g_loss = self.combined.train_on_batch([gt_img,input_img],[valid,gt_img])
				elapsed_time=datetime.datetime.now()-start_time
				g_avg = g_avg + g_loss[0]
				d_avg = d_avg + 100*d_loss[1]
				count = count + 1
			now = datetime.datetime.now()
			system('clear')
			print('Training...')
			print('Time: %s'%(elapsed_time))
			print('[Epoch %d/%d]'%(epoch+1,epochs))
			print('[G Loss: %d]'%(g_avg/count))
			print('[D Accuracy: %d]'%(d_avg/count))
			print('[Timestamp: %s]'%(str(now.strftime('%m/%d/%Y %H:%M:%S'))))
			prev_elapsed_time = elapsed_time
			if (epoch+1) % self.sample_interval == 0:
				stokes_image = tf.clip_by_value(tf.cast(tf.signal.ifft2d(tf.complex(gt_img[0,:,:,0],gt_img[0,:,:,1])),dtype=tf.float64),0,2**16).numpy()
				self.fs0_gt2.set_data(self.imgscale(stokes_image,stype="statistical"))
				self.fgt2.canvas.flush_events()
				stokes_image = tf.clip_by_value(tf.cast(tf.signal.ifft2d(tf.complex(gen_img[0,:,:,0],gen_img[0,:,:,1])),dtype=tf.float64),0,2**16).numpy()
				self.fs0_gen2.set_data(self.imgscale(stokes_image,stype="statistical"))
				self.fgen2.canvas.flush_events()
				self.fs0_gt.set_data(self.imgscale(np.log10(1+np.abs(tf.signal.fftshift(tf.complex(gt_img[0,:,:,0],gt_img[0,:,:,1])).numpy())),stype="statistical"))
				self.fgt.canvas.flush_events()
				self.fs0_gen.set_data(self.imgscale(np.log10(1+np.abs(tf.signal.fftshift(tf.complex(gen_img[0,:,:,0],gen_img[0,:,:,1])).numpy())),stype="statistical"))
				self.fgen.canvas.flush_events()
				self.save_model(epoch+1)

	def save_model(self,epoch):
		filename='%s%s'%(self.saveFolder,self.model)
		self.generator.save(filename)

	def test(self):
		b=12
		plt.ion()
		self.fgt = plt.figure()
		self.fs0_gt = plt.imshow(np.random.rand(2048*2496).reshape((2048,2496)),cmap='gray')
		plt.title('GT S0')
		self.fgt.canvas.flush_events()
		self.fgen = plt.figure()
		self.fs0_gen = plt.imshow(np.random.rand(2048*2496).reshape((2048,2496)),cmap='gray')
		plt.title('Gen S0')
		self.fgen.canvas.flush_events()
		for batch_i, (full_image,gt_img,ugrid_name,stokes_name) in enumerate(self.load_test_data()):
			count = 1
			print('')
			print('[Test Image %s]' % (ugrid_name))
			full_image_channels=1
			full_image_rows,full_image_cols = full_image.shape
			print('[Image Rows %s]' % (str(full_image_rows)))
			print('[Image Cols %s]' % (str(full_image_cols)))
			print('[Image Depth %s]' % (str(full_image_channels)))
			sub_image_rows = self.img_rows
			sub_image_cols = self.img_cols
			row_iterations = (np.floor(full_image_rows/(sub_image_rows-(2*b+3)))).astype(np.int)*2
			col_iterations = (np.floor(full_image_cols/(sub_image_cols-(2*b+3)))).astype(np.int)*2
			print('[Row Iterations %s]' % (str(row_iterations)))
			print('[Col Iterations %s]' % (str(col_iterations)))
			print('[Overlap %s]' % (str(b)))
			image_count = 1
			print('[Processing Sub Images %d of %d]' % (count,row_iterations*col_iterations))
			sub_image = np.zeros((1,sub_image_rows,sub_image_cols,self.input_channels),dtype=np.float64)
			tmp = np.zeros((sub_image_rows,sub_image_cols),dtype=np.float64)
			gen_image = np.zeros((1,sub_image_rows,sub_image_cols,self.output_channels),dtype=np.float64)
			stokes_image = np.zeros((full_image_rows,full_image_cols,3),dtype=np.float64)
			ys=0
			ye=0
			breakoutflag=0
			for r in range(row_iterations+1):
				if (breakoutflag):
					break
				xs=0
				xe=0
				if r==0:
					ys=0
					ye = sub_image_rows-1
				else:
					ys = ye -3 -2*b
					ye = ys + sub_image_rows - 1
				for c in range(col_iterations+1):
					if c==0:
						xs = 0
						xe = sub_image_cols - 1
					else:
						xs = xe - 3 -2*b
						xe = xs + sub_image_cols -1
					if ye >= full_image_rows:
						if xe >= full_image_cols:
							ye = full_image_rows-1
							ys = ye - sub_image_rows+1
							xe = full_image_cols-1
							xs = xe - sub_image_cols+1
							x = tf.cast(tf.reduce_sum(tf.abs(full_image[ys:ye+1,xs:xe+1])),tf.complex128)
							tmp = tf.signal.fft2d(tf.cast(full_image[ys:ye+1,xs:xe+1],tf.complex128))/x
							sub_image[0,:,:,0] = tf.cast(tf.math.real(tmp),dtype=tf.float64)
							sub_image[0,:,:,1] = tf.cast(tf.math.imag(tmp),dtype=tf.float64)
							gen_image[:,:,:,:] = self.generator.predict(sub_image)
							stokes_image[ys:ye+1,xs:xe+1,0] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,0],gen_image[0,:,:,1])*x),dtype=tf.float64).numpy()
							stokes_image[ys:ye+1,xs:xe+1,1] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,2],gen_image[0,:,:,3])*x),dtype=tf.float64).numpy()
							stokes_image[ys:ye+1,xs:xe+1,2] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,4],gen_image[0,:,:,5])*x),dtype=tf.float64).numpy()
							breakoutflag=1
							break
						else:
							ye = full_image_rows-1
							ys = ye - sub_image_rows + 1
					if xe >= full_image_cols:
						xe = full_image_cols-1
						xs = xe - sub_image_cols+1
						x = tf.cast(tf.reduce_sum(tf.abs(full_image[ys:ye+1,xs:xe+1])),tf.complex128)
						tmp = tf.signal.fft2d(tf.cast(full_image[ys:ye+1,xs:xe+1],tf.complex128))/x
						sub_image[0,:,:,0] = tf.cast(tf.math.real(tmp),dtype=tf.float64)
						sub_image[0,:,:,1] = tf.cast(tf.math.imag(tmp),dtype=tf.float64)
						gen_image[:,:,:,:] = self.generator.predict(sub_image)
						stokes_image[ys:ye+1,xs:xe+1,0] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,0],gen_image[0,:,:,1])*x),dtype=tf.float64).numpy()
						stokes_image[ys:ye+1,xs:xe+1,1] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,2],gen_image[0,:,:,3])*x),dtype=tf.float64).numpy()
						stokes_image[ys:ye+1,xs:xe+1,2] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,4],gen_image[0,:,:,5])*x),dtype=tf.float64).numpy()
						if (ye == full_image_rows-1):
							breakoutflag=1
						break
					x = tf.cast(tf.reduce_sum(tf.abs(full_image[ys:ye+1,xs:xe+1])),tf.complex128)
					tmp = tf.signal.fft2d(tf.cast(full_image[ys:ye+1,xs:xe+1],tf.complex128))/x
					sub_image[0,:,:,0] = tf.cast(tf.math.real(tmp),dtype=tf.float64)
					sub_image[0,:,:,1] = tf.cast(tf.math.imag(tmp),dtype=tf.float64)
					if (r==0):
						if (c==0):
							gen_image[:,:,:,:] = self.generator.predict(sub_image)
							stokes_image[ys:ye+1,xs:xe+1,0] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,0],gen_image[0,:,:,1])*x),dtype=tf.float64).numpy()
							stokes_image[ys:ye+1,xs:xe+1,1] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,2],gen_image[0,:,:,3])*x),dtype=tf.float64).numpy()
							stokes_image[ys:ye+1,xs:xe+1,2] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,4],gen_image[0,:,:,5])*x),dtype=tf.float64).numpy()
						else:
							gen_image[:,:,:,:] = self.generator.predict(sub_image)
							stokes_image[ys:ye+1,xs+b:xe+1,0] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,0],gen_image[0,:,:,1])*x)[:,b::],dtype=tf.float64).numpy()
							stokes_image[ys:ye+1,xs+b:xe+1,1] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,2],gen_image[0,:,:,3])*x)[:,b::],dtype=tf.float64).numpy()
							stokes_image[ys:ye+1,xs+b:xe+1,2] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,4],gen_image[0,:,:,5])*x)[:,b::],dtype=tf.float64).numpy()
					elif (c==0):
						gen_image[:,:,:,:] = self.generator.predict(sub_image)
						stokes_image[ys+b:ye+1,xs:xe+1,0] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,0],gen_image[0,:,:,1])*x)[b::,:],dtype=tf.float64).numpy()
						stokes_image[ys+b:ye+1,xs:xe+1,1] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,2],gen_image[0,:,:,3])*x)[b::,:],dtype=tf.float64).numpy()
						stokes_image[ys+b:ye+1,xs:xe+1,2] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,4],gen_image[0,:,:,5])*x)[b::,:],dtype=tf.float64).numpy()
					else:
						gen_image[:,:,:,:] = self.generator.predict(sub_image)
						stokes_image[ys+b:ye+1,xs+b:xe+1,0] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,0],gen_image[0,:,:,1])*x)[b::,b::],dtype=tf.float64).numpy()
						stokes_image[ys+b:ye+1,xs+b:xe+1,1] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,2],gen_image[0,:,:,3])*x)[b::,b::],dtype=tf.float64).numpy()
						stokes_image[ys+b:ye+1,xs+b:xe+1,2] = tf.cast(tf.signal.ifft2d(tf.complex(gen_image[0,:,:,4],gen_image[0,:,:,5])*x)[b::,b::],dtype=tf.float64).numpy()
					system('clear')
					print('[Test Image %s]' % (ugrid_name))
					print('[Image Rows %s]' % (str(full_image_rows)))
					print('[Image Cols %s]' % (str(full_image_cols)))
					print('[Image Depth %s]' % (str(full_image_channels)))
					print('[Row Iterations %d of %s]' % (r,str(row_iterations)))
					print('[Col Iterations %d of %s]' % (c,str(col_iterations)))
					print('[Overlap %s]' % (str(b)))
					print('[Processing Sub Images %d of %d]' % (count,row_iterations*col_iterations))
					count = count + 1
			save_name = '%sgenerated_in_fft_out_stokes_%s'	% (saveFolder,stokes_name)
			sio.savemat(save_name,{'stokes' : stokes_image.astype(np.float64)})
			image_count = image_count + 1	
			self.fs0_gt.set_data(self.imgscale(gt_img[:,:,0],stype="statistical"))
			self.fgt.canvas.flush_events()
			self.fs0_gen.set_data(self.imgscale(stokes_image[:,:,0].astype(np.float64),stype="statistical"))
			self.fgen.canvas.flush_events()
			
					

if __name__ == '__main__':
	epochs= 2000 
	modelNumber = 240 
	dataFolder = '/app/data/'
	saveFolder = '/app/data/results/'
	modelFolder = '/app/data/results/'
	#model='model_%d.h5'%(modelNumber)
	model='model_fft_32.h5'
	comments = ''
	
	gan_train = MicrogridGAN(mode=1,dataFolder=dataFolder,saveFolder=saveFolder,modelFolder=modelFolder,model=model,comments=comments)
	input('Press any key to continue...')
	#gan_train.test()
	#gan_test = MicrogridGAN(mode=2,dataFolder=dataFolder,saveFolder=saveFolder,modelFolder=modelFolder,model=model,comments=comments)
	#input('Press any key to continue...')
