import tensorflow as tf
import math as m
import math

class HelperFunctions():

	@staticmethod
	def mae_loss_fft(target,gen_output):
		pi = tf.constant(m.pi)
		#s0_target = tf.cast(tf.signal.ifft2d(tf.cast(tf.complex(target[0,:,:,0],target[0,:,:,1]),dtype=tf.complex64)),tf.float32)
		#s0_gen = tf.clip_by_value(tf.cast(tf.signal.ifft2d(tf.cast(tf.complex(gen_output[0,:,:,0],gen_output[0,:,:,1]),dtype=tf.complex64)),tf.float32),0,2**17-1)
		#s1_target = tf.cast(tf.signal.ifft2d(tf.cast(tf.complex(target[0,:,:,2],target[0,:,:,3]),dtype=tf.complex64)),tf.float32)
		#s1_gen = tf.cast(tf.signal.ifft2d(tf.cast(tf.complex(gen_output[0,:,:,2],gen_output[0,:,:,3]),dtype=tf.complex64)),tf.float32)
		#s2_target = tf.cast(tf.signal.ifft2d(tf.cast(tf.complex(target[0,:,:,4],target[0,:,:,5]),dtype=tf.complex64)),tf.float32)
		#s2_gen = tf.cast(tf.signal.ifft2d(tf.cast(tf.complex(gen_output[0,:,:,4],gen_output[0,:,:,5]),dtype=tf.complex64)),tf.float32)

		#I0 = 0.5*(s0_gen + tf.math.cos(2*(0)*(pi/180))*s1_gen + tf.math.sin(2*(0)*(pi/180))*s2_gen)
		#I30 = 0.5*(s0_gen + tf.math.cos(2*(30)*(pi/180))*s1_gen + tf.math.sin(2*(30)*(pi/180))*s2_gen)
		#I60 = 0.5*(s0_gen + tf.math.cos(2*(60)*(pi/180))*s1_gen + tf.math.sin(2*(60)*(pi/180))*s2_gen)
		#I90 = 0.5*(s0_gen + tf.math.cos(2*(90)*(pi/180))*s1_gen + tf.math.sin(2*(90)*(pi/180))*s2_gen)
		#I120 = 0.5*(s0_gen + tf.math.cos(2*(120)*(pi/180))*s1_gen + tf.math.sin(2*(120)*(pi/180))*s2_gen)
		#I150 = 0.5*(s0_gen + tf.math.cos(2*(150)*(pi/180))*s1_gen + tf.math.sin(2*(150)*(pi/180))*s2_gen)

		#r1=tf.reduce_mean(tf.abs(I0+I90-I30-I120))
		#r2=tf.reduce_mean(tf.abs(I0+I90-I60-I150))
		#r3=tf.reduce_mean(tf.abs(I30+I120-I60-I150))
		#r4=tf.reduce_mean(tf.abs(I0-I90-2*I30+2*I60))
		#r5=tf.reduce_mean(tf.abs(I0-I90-2*I150+2*I120))
		#r6=tf.reduce_mean(tf.abs(I30-I60-I150+I120))
        
		fft_s0_loss_real = tf.reduce_mean(tf.abs(target[:,:,:,0]-gen_output[:,:,:,0]))
		fft_s0_loss_imag = tf.reduce_mean(tf.abs(target[:,:,:,1]-gen_output[:,:,:,1]))
		fft_s1_loss_real = tf.reduce_mean(tf.abs(target[:,:,:,2]-gen_output[:,:,:,2]))
		fft_s1_loss_imag = tf.reduce_mean(tf.abs(target[:,:,:,3]-gen_output[:,:,:,3]))
		fft_s2_loss_real = tf.reduce_mean(tf.abs(target[:,:,:,4]-gen_output[:,:,:,4]))
		fft_s2_loss_imag = tf.reduce_mean(tf.abs(target[:,:,:,5]-gen_output[:,:,:,5]))
		fft_loss = fft_s0_loss_real + fft_s0_loss_imag + fft_s1_loss_real + fft_s1_loss_imag + fft_s2_loss_real + fft_s2_loss_imag
		#fft_loss = fft_s0_loss_real #+ fft_s0_loss_imag
		#s0_loss = tf.reduce_mean(tf.abs(s0_target-s0_gen))
		#s1_loss = tf.reduce_mean(tf.abs(s1_target-s1_gen))
		#s2_loss = tf.reduce_mean(tf.abs(s2_target-s2_gen))
		#stokes_loss = s0_loss + s1_loss + s2_loss
		#return tf.cast(fft_loss + s0_loss + s1_loss + s2_loss,dtype=tf.float64) 
		total_loss = fft_loss
		return total_loss 

	@staticmethod
	def mae_loss_stokes(target,gen_output):
		pi = tf.constant(m.pi)
		s0_gen = tf.clip_by_value(gen_output[:,:,:,0],0,2**17-1)
		s1_gen = gen_output[:,:,:,1]
		s2_gen = gen_output[:,:,:,2]
		s0_target = target[:,:,:,0]
		s1_target = target[:,:,:,1]
		s2_target = target[:,:,:,2]
		I0 = 0.5*(s0_gen + tf.math.cos(2*(0)*(pi/180))*s1_gen + tf.math.sin(2*(0)*(pi/180))*s2_gen)
		I30 = 0.5*(s0_gen + tf.math.cos(2*(30)*(pi/180))*s1_gen + tf.math.sin(2*(30)*(pi/180))*s2_gen)
		I60 = 0.5*(s0_gen + tf.math.cos(2*(60)*(pi/180))*s1_gen + tf.math.sin(2*(60)*(pi/180))*s2_gen)
		I90 = 0.5*(s0_gen + tf.math.cos(2*(90)*(pi/180))*s1_gen + tf.math.sin(2*(90)*(pi/180))*s2_gen)
		I120 = 0.5*(s0_gen + tf.math.cos(2*(120)*(pi/180))*s1_gen + tf.math.sin(2*(120)*(pi/180))*s2_gen)
		I150 = 0.5*(s0_gen + tf.math.cos(2*(150)*(pi/180))*s1_gen + tf.math.sin(2*(150)*(pi/180))*s2_gen)

		r1=tf.reduce_mean(tf.abs(I0+I90-I30-I120))
		r2=tf.reduce_mean(tf.abs(I0+I90-I60-I150))
		r3=tf.reduce_mean(tf.abs(I30+I120-I60-I150))
		r4=tf.reduce_mean(tf.abs(I0-I90-2*I30+2*I60))
		r5=tf.reduce_mean(tf.abs(I0-I90-2*I150+2*I120))
		r6=tf.reduce_mean(tf.abs(I30-I60-I150+I120))
        
		s0_loss = tf.reduce_mean(tf.abs(s0_target-s0_gen))
		s1_loss = tf.reduce_mean(tf.abs(s1_target-s1_gen))
		s2_loss = tf.reduce_mean(tf.abs(s2_target-s2_gen))
		stokes_loss = s0_loss + s1_loss + s2_loss

		total_loss = stokes_loss + r1 + r2 + r3 + r4 + r5 + r6
		return total_loss 

	@staticmethod
	def generator_mae_loss_stokes(target,gen):
		s0_loss = tf.reduce_mean(tf.abs(target[:,:,:,0]-gen[:,:,:,0]))
		s1_loss = tf.reduce_mean(tf.abs(target[:,:,:,1]-gen[:,:,:,1]))
		s2_loss = tf.reduce_mean(tf.abs(target[:,:,:,2]-gen[:,:,:,2]))
		stokes_loss = s0_loss + s1_loss + s2_loss
		return stokes_loss


	@staticmethod
	def generator_mae_loss(target,gen_output):
		total_gen_loss = tf.reduce_mean(tf.abs(target-gen_output))
		return total_gen_loss
 
	@staticmethod
	def pad3(tensor):
		pad1=[1,1] #[1,1] top-bottom
		pad2=[1,1] #[1,1] left-right
		return tf.pad(tensor, [[0,0], pad1, pad2, [0,0] ], mode='SYMMETRIC')
        
	@staticmethod    
	def pad4(tensor):
		pad1=[1,2]
		pad2=[1,2]
		return tf.pad(tensor, [[0,0], pad1, pad2, [0,0] ], mode='SYMMETRIC')
    
	@staticmethod
	def pad5(tensor):
		pad1=[2,2]
		pad2=[2,2]
		return tf.pad(tensor, [[0,0], pad1, pad2, [0,0] ], mode='SYMMETRIC')
    
	@staticmethod   
	def padmebaby(tensor):
		pad1=[1,1]
		pad2=[1,1]
		# pad1=[0,1]
		# pad2=[0,1]
		return tf.pad(tensor, [[0,0], pad1, pad2, [0,0] ], mode='SYMMETRIC')
    
	@staticmethod
	def padmeharderbaby(tensor):
		pad1=[1,2]
		pad2=[1,2]
		# pad1=[1,2]
		# pad2=[1,2]
		return tf.pad(tensor, [[0,0], pad1, pad2, [0,0] ], mode='SYMMETRIC') 
