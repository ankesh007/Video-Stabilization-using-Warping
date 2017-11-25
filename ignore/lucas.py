import numpy as np
import cv2
from scipy.ndimage import convolve

image1=cv2.imread('corners.jpg')
image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2=cv2.imread('movement.jpg')
image2=cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
window_size=25
error_val=0.1

shape_image=np.shape(image2)
parameters=6

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

features = cv2.goodFeaturesToTrack(image2, mask = None, **feature_params)
features=features[:,0,:]
# print features.shape
# exit(1)

p=np.zeros([parameters,1], dtype=np.float)
p[0]=1
p[3]=1
# print p

indices=np.indices(shape_image)
indices=np.transpose(indices,[1,2,0])
jacobian_wrap=np.zeros([shape_image[0],shape_image[1],2,6])
jacobian_wrap[:,:,0,0]=indices[:,:,1]
jacobian_wrap[:,:,0,2]=indices[:,:,0]
jacobian_wrap[:,:,0,4]=1
jacobian_wrap[:,:,1,1:parameters]=jacobian_wrap[:,:,0,0:parameters-1]
# print jacobian_wrap.shape
coun=0

point_to_transform=np.float32([[50,50],[50,200],[200,50]])
new_point=np.zeros([3,2],dtype=np.float32)
# print point_to_transform.shape
# exit(1)
# transformed_points=np.

def get_new_points():
	new_point[:,0]=point_to_transform[:,0]*p[0]+point_to_transform[:,1]*p[2]+p[4]
	new_point[:,1]=point_to_transform[:,0]*p[1]+point_to_transform[:,1]*p[3]+p[5]


centrej=int(features[0][1])
centrei=int(features[0][0])
extension=window_size//2
jacobian_wrap=jacobian_wrap[max(0,centrei-extension):min(shape_image[0],centrei+extension+1),max(0,centrej-extension):min(shape_image[1],centrej+extension+1)]

while(True):

	get_new_points()
	M=cv2.getAffineTransform(point_to_transform,new_point)
	warped_image=cv2.warpAffine(image2,M,(shape_image[1],shape_image[0]))

	error=image1-warped_image
	# print centrei
	# print centrej


	gradientx,gradienty=np.gradient(image2)
	# gradientx = cv2.Sobel(image2,cv2.CV_64F,1,0,ksize=3)
	# gradienty = cv2.Sobel(image2,cv2.CV_64F,0,1,ksize=3)
	warped_gradx=cv2.warpAffine(gradientx,M,(shape_image[1],shape_image[0]))
	warped_grady=cv2.warpAffine(gradienty,M,(shape_image[1],shape_image[0]))
	gradient=np.stack((warped_gradx,warped_grady),axis=2)
	error=error[max(0,centrei-extension):min(shape_image[0],centrei+extension+1),max(0,centrej-extension):min(shape_image[1],centrej+extension+1)]
	gradient=gradient[:,:,np.newaxis,:]
	# print gradient.shape -img*img*1*2
	gradient=gradient[max(0,centrei-extension):min(shape_image[0],centrei+extension+1),max(0,centrej-extension):min(shape_image[1],centrej+extension+1),:,:]
	# win*win*2*6
	# print gradient
	steepest_descent=np.matmul(gradient,jacobian_wrap)
	steepest_descent_transpose=np.transpose(steepest_descent,[0,1,3,2])
	# print (np.linalg.inv(steepest_descent))
	# exit(1)
	# print steepest_descent.shape -window*window*1*6
	# aux_for_hessian=np.zeros([window_size,window_size,parameters,parameters])

	# for i in range(window_size):
	# 	for j in range(window_size):
	# 		aux_for_hessian[i,j,:,:]=np.matmul(steepest_descent_transpose[i,j,:,:],steepest_descent[:,:,i,j])

	aux_for_hessian=np.matmul(steepest_descent_transpose,steepest_descent)

	# print aux_for_hessian
	# exit(1)
	# print aux_for_hessian.shape win*win*para*para
	hessian=np.zeros([parameters,parameters])
	hessian=np.sum(np.sum(aux_for_hessian,axis=0),axis=0)
	# print hessian
	# exit(1)
	# print hessian.shape win*win
	temp_steepest_grad_error=np.matmul(steepest_descent_transpose,error[:,:,np.newaxis,np.newaxis])
	# print temp_steepest_grad_error.shape win*win*parameters*1

	before_delp=np.zeros([6,1])
	before_delp=np.sum(np.sum(temp_steepest_grad_error,axis=0),axis=0)

	# print before_delp.shape

	# print hessian.shape

	hessian_inv=np.linalg.inv(hessian)
	# print "Hi"
	delp=np.matmul(hessian_inv,before_delp)

	# print delp.shape
	# print "Hi"

	delp_dimred=np.linalg.norm(delp,axis=0)
	# print delp_dimred
	# print error.shape

	if error_val>delp_dimred:
		break
	# mask=delp_dimred>error

	p=p+delp
	coun=coun+1
	print coun
	print delp_dimred
	# print delp

	if(coun%100==0):
		cv2.imwrite("yo.jpg",warped_image)
