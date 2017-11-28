from moviepy.editor import VideoFileClip
import numpy as np
import cv2
from scipy.ndimage import convolve
import sys


input_video_path=(sys.argv[1])
# print input_video_path
# exit(1)
# input_video_path=0
output_video_path=sys.argv[2]
cap = cv2.VideoCapture(input_video_path)
ret,target=cap.read()
length=target.shape[0]/4
breadth=target.shape[1]/4
fps=10
# print target.shape
target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target=cv2.resize(target,(breadth,length))

def get_points(img):
    points = []
    img_to_show = img.copy()
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_to_show,(x,y),2,(255,0,0),-1)
            points.append([x,y])
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img_to_show)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return points

points_on_image=get_points(target)
target = target.astype('float32')
target=target[points_on_image[0][1]:points_on_image[1][1],points_on_image[0][0]:points_on_image[1][0]]
shape_image=target.shape
# print shape_image
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(output_video_path+'.avi',fourcc, fps, (breadth,length))
# print target.shape
process_after=1
error_val=0.005
iteration_limit=2500
parameters=6
# image shape 720 X 1280

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
# print hessian_inv
# print hessian.shape parameter*parameter

# **********Precomutation Ends******************

def warper(image):

	p=np.zeros([parameters,1], dtype=np.float)
	p[0]=1
	p[3]=1
	source=np.copy(image)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
	# source=cv2.resize(source,(length,breadth))
	coun=0

	while(True):

		coun=coun+1
		warped_image = cv2.warpAffine(source, np.transpose(np.reshape(p,[3,2])), (target.shape[1],target.shape[0]))

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
		# print delp_dimred
		if error_val>delp_dimred:
			return p
		if coun>iteration_limit:
			return p

			

frame_count=0
while(1):
	r,frame=cap.read()

	frame_count=frame_count+1
	if(frame_count%process_after!=0):
		print frame_count
		continue
	else:
		print frame_count
		# print "processing"

	if(frame is None):
		break

	frame=cv2.resize(frame,(breadth,length))
	frame_parameter=warper(frame[points_on_image[0][1]:points_on_image[1][1],points_on_image[0][0]:points_on_image[1][0]])
	frame_parameter[4]=frame_parameter[4]+points_on_image[0][0]-frame_parameter[0]*points_on_image[0][0]-frame_parameter[2]*points_on_image[0][1]
	frame_parameter[5]=frame_parameter[5]+points_on_image[0][1]-frame_parameter[1]*points_on_image[0][0]-frame_parameter[3]*points_on_image[0][1]
	
	frame=cv2.warpAffine(frame, np.transpose(np.reshape(frame_parameter,[3,2])), (frame.shape[1],frame.shape[0]))
	frame.astype(np.uint8)
	out.write(frame)
	cv2.imshow("image",frame)


	kk = cv2.waitKey(60) & 0xff
	if kk == 27:
		break

cap.release()
out.release()
cv2.destroyAllWindows()
