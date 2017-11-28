import numpy as np
import cv2
from scipy.ndimage import convolve
import sys

if(len(sys.argv)<5):
	print "Usage:<filename> <source image path> <target image path> <epsilon> <output_path>"
	exit(1)

source=cv2.imread(sys.argv[1])
source=cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
target=cv2.imread(sys.argv[2])
target=cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
scale=int(target.shape[0]/300)
length=target.shape[0]/scale
breadth=target.shape[1]/scale
source=cv2.resize(source,(breadth,length))
target=cv2.resize(target,(breadth,length))
source = source.astype('float32')
target = target.astype('float32')

shape_image=source.shape
error_val=float(sys.argv[3])
iteration_limit=2500
parameters=6
# image shape 720 X 1280

p=np.zeros([parameters,1], dtype=np.float)
p[0]=1
p[3]=1
# print p

# ********not required**********
# point_to_transform=np.float32([[50,50],[50,200],[200,50]])
# new_point=np.zeros([3,2],dtype=np.float32)
# def get_new_points():
# 	global new_point
# 	new_point[:,0]=point_to_transform[:,0]*p[0]+point_to_transform[:,1]*p[2]+p[4]
# 	new_point[:,1]=point_to_transform[:,0]*p[1]+point_to_transform[:,1]*p[3]+p[5]

def update_p(p,del_p):
	second=np.zeros([parameters,1], dtype=np.float)
	third=np.zeros([parameters,1], dtype=np.float)
	second[0]=p[0]*del_p[0]
	second[1]=p[1]*del_p[0]
	second[2]=p[0]*del_p[2]
	second[3]=p[1]*del_p[2]
	second[4]=p[0]*del_p[4]
	second[5]=p[1]*del_p[4]
	third[0]=p[2]*del_p[1]
	third[1]=p[3]*del_p[1]
	third[2]=p[2]*del_p[3]
	third[3]=p[3]*del_p[3]
	third[4]=p[2]*del_p[5]
	third[5]=p[3]*del_p[5]
	return p+del_p+second+third

def update_p_using_inverse(p,del_p):
	second=np.zeros([parameters,1], dtype=np.float)
	third=np.zeros([parameters,1], dtype=np.float)
	second[0]=del_p[0]*del_p[3]
	second[3]=del_p[0]*del_p[3]
	second[4]=del_p[3]*del_p[4]
	second[5]=del_p[0]*del_p[5]
	third[0]=del_p[1]*del_p[2]
	third[3]=del_p[1]*del_p[2]
	third[4]=del_p[2]*del_p[5]
	third[5]=del_p[1]*del_p[4]

	return update_p(p,-del_p-second+third)
	


# **********Precomutation******************
indices=np.indices(shape_image)
indices=np.transpose(indices,[1,2,0])
jacobian_wrap=np.zeros([shape_image[0],shape_image[1],2,6],)
jacobian_wrap[:,:,0,0]=indices[:,:,1]
jacobian_wrap[:,:,0,2]=indices[:,:,0]
jacobian_wrap[:,:,0,4]=1
jacobian_wrap[:,:,1,1:parameters]=np.copy(jacobian_wrap[:,:,0,0:parameters-1])
coun=0


# del_tx,del_ty=np.gradient(target)
del_tx = cv2.Sobel(target, cv2.CV_32F, 1, 0)
del_ty = cv2.Sobel(target, cv2.CV_32F, 0, 1)
# print del_tx.shape(720, 1280)

del_t=np.stack((del_tx,del_ty),axis=2)
del_t=del_t[:,:,np.newaxis,:]
steepest_descent=np.matmul(del_t,jacobian_wrap)
# print steepest_descent.shape (720, 1280, 1, 6)

steepest_descent_transpose=np.transpose(steepest_descent,(0,1,3,2))
# print steepest_descent_transpose.shape
hessian=np.matmul(steepest_descent_transpose,steepest_descent)
hessian=np.sum(np.sum(hessian,axis=0),axis=0)
hessian_inv=np.linalg.inv(hessian)
# print hessian.shape parameter*parameter

# **********Precomutation Ends******************

# def warper(image)

while(True):

	coun=coun+1
	warped_image = cv2.warpAffine(source, np.transpose(np.reshape(p,[3,2])), (target.shape[1],target.shape[0]))

	# cv2.imwrite("Image/hi"+str(coun)+".jpg",warped_image)

	error=warped_image-target

	# print error.shape 720 X 1280
	aux_val=np.matmul(steepest_descent_transpose,error[:,:,np.newaxis,np.newaxis])
	aux_val=np.sum(np.sum(aux_val,axis=0),axis=0)
	# print aux_val.shape 6(parameter) X 1

	del_p=np.matmul(hessian_inv,aux_val)
	# print del_p
	# print dkiel_p.shape (6X1)

	p=update_p(p,del_p)
	# p=update_p_using_inverse(p,del_p)

	delp_dimred=np.linalg.norm(del_p,axis=0)
	print coun,delp_dimred

	if error_val>delp_dimred:
		cv2.imwrite(sys.argv[4],np.concatenate((source,warped_image,target),axis=1))
		break
	if coun>iteration_limit:
		cv2.imwrite(sys.argv[4],np.concatenate((source,warped_image,target),axis=1))
		break
