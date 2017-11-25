import numpy as np
import cv2
from scipy.ndimage import convolve
import msvcrt as m
# cap = cv2.VideoCapture('vtest.avi')

image1=np.zeros([20,20])
window_size=15
image2=np.zeros([20,20])
error=0.1
# image1=cv2.imread('Warp2.jpg')
# image1=cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

shape_image=np.shape(image1)

parameters=6
warp=np.zeros([shape_image[0],shape_image[1],2],dtype=np.int)

p=np.zeros([shape_image[0],shape_image[1],parameters], dtype=np.float)
p[:,:,0]=1
p[:,:,3]=1

indices=np.indices(shape_image)
indices=np.transpose(indices,[1,2,0])

jacobian_wrap=np.zeros([shape_image[0],shape_image[1],2,6])
jacobian_wrap[:,:,0,0]=indices[:,:,0]
jacobian_wrap[:,:,0,2]=indices[:,:,1]
jacobian_wrap[:,:,0,4]=1
jacobian_wrap[:,:,1,1:parameters]=jacobian_wrap[:,:,0,0:parameters-1]
coun=0

def calc_warp():
	global warp
	warp[:,:,0]=p[:,:,0]*indices[:,:,0] + p[:,:,2]*indices[:,:,1] + p[:,:,4]
	warp[:,:,1]=p[:,:,1]*indices[:,:,0] + p[:,:,3]*indices[:,:,1] + p[:,:,5]

	mask1=warp[:,:,0]>=shape_image[0]
	warp[:,:,0]=warp[:,:,0]*(1-mask1)+mask1*(shape_image[0]-1)
	mask1=warp[:,:,1]>=shape_image[1]
	warp[:,:,1]=warp[:,:,1]*(1-mask1)+mask1*(shape_image[1]-1)
	mask1=warp[:,:,0]<0
	warp[:,:,0]=warp[:,:,0]*(1-mask1)
	mask1=warp[:,:,1]<0
	warp[:,:,1]=warp[:,:,1]*(1-mask1)


while (True):
	calc_warp()
	warped_image=np.zeros([shape_image[0],shape_image[1]])
	warped_image=image1[warp[:,:,0],warp[:,:,1]]

	error=image2-warped_image
	gradientx,gradienty=np.gradient(image1)
	warped_gradx=gradientx[warp[:,:,0],warp[:,:,1]]
	warped_grady=gradienty[warp[:,:,0],warp[:,:,1]]

	gradient=np.stack((warped_gradx,warped_grady),axis=2)
	gradient=gradient[:,:,np.newaxis,:]

	steepest_descent=np.matmul(gradient,jacobian_wrap)
	steepest_descent_transpose=np.transpose(steepest_descent,[0,1,3,2])

	aux_for_hessian=np.matmul(steepest_descent_transpose,steepest_descent)
	# parameters=
	filter_2d=np.ones([window_size,window_size,6,6])
	hessian=convolve(aux_for_hessian,filter_2d)
	temp_steepest_grad_error=np.matmul(steepest_descent_transpose,error[:,:,np.newaxis,np.newaxis])
	# print temp_steepest_grad_error.shape

	filter_2d_2=np.ones([window_size,window_size,6,1])
	before_delp=convolve(temp_steepest_grad_error,filter_2d_2)

	hessian_inv=np.linalg.inv(hessian)
	delp=np.matmul(hessian_inv,before_delp)[:,:,:,0]

	delp_dimred=np.linalg.norm(delp,axis=2)
	mask=delp_dimred>error

	delp=delp+mask*delp_dimred
	coun=coun+1

	if(coun%100==0)
		cv2.imwrite(warped_image,"yo.jpg")
		m.getch()

	mask=np.sum(mask)

	if(mask==0):
		break

# print delp.shape

# delp.shape
# print steepest_descent.shape
# print jacobian_wrap
# print gradientx.shape


# cv2.imwrite('yo.jpg',gradientx)
# cv2.imwrite('yo2.jpg',gradienty)



# print warped_image.shape
# warped_image[indices[:,:,0],indices[:,:,1]]=image1[warp[indices[:,:,0],indices[:,:,1],0],warp[indices[:,:,0],indices[:,:,1],1]]

