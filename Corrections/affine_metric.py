import numpy as np
import cv2
import sys

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

def inhomogenize(point):
    if(point[2]!=0):
        point[0]=point[0]/point[2]
        point[1]=point[1]/point[2]
        point[2]=point[2]/point[2]

def getVanishingPoint(line1,line2):
    vanishing_point=np.cross(line1,line2)
    inhomogenize(vanishing_point)
    return vanishing_point

def get_line(p1,p2):
    line=np.cross(p1,p2)
    inhomogenize(line)
    return line

def affine_correct(input_image,points_on_image):

    # points_on_image=get_points(input_image)
    print points_on_image
    if(len(points_on_image)<4):
        print "Incomplete Selection"
        return input_image

    points_on_image=np.concatenate((points_on_image,np.ones([4,1])),axis=1)
    numpy_list=np.asarray(points_on_image,dtype=np.float32)

    parallel_line1_1=get_line(numpy_list[0],numpy_list[2])
    parallel_line1_2=get_line(numpy_list[1],numpy_list[3])
    vanishing_point_1=getVanishingPoint(parallel_line1_1,parallel_line1_2)

    parallel_line2_1=get_line(numpy_list[0],numpy_list[1])
    parallel_line2_2=get_line(numpy_list[2],numpy_list[3])
    vanishing_point_2=getVanishingPoint(parallel_line2_1,parallel_line2_2)

    vanishing_line=get_line(vanishing_point_1,vanishing_point_2)
    print vanishing_line
    affine_correct_projection=np.array([[1,0,0],[0,1,0],vanishing_line])
    destination=cv2.warpPerspective(input_image,affine_correct_projection,(input_image.shape[1],input_image.shape[0]))
    return destination
    
def metric_correction(image,refPt):

	pt1=np.asarray(refPt,dtype=np.float32)
	dist=(refPt[1][0]-refPt[0][0])
	refPt[1]=(refPt[0][0]+dist,refPt[0][1])
	refPt[2]=(refPt[0][0],refPt[0][1]+dist)
	refPt[3]=(refPt[0][0]+dist,refPt[0][1]+dist)
	pt2=np.asarray(refPt,dtype=np.float32)
	M=cv2.getPerspectiveTransform(pt1,pt2)
	dst=cv2.warpPerspective(image,M,(image.shape[1],image.shape[0]))
	return dst

def main():
    input_image_path=sys.argv[1]
    image=cv2.imread(input_image_path)
    points=get_points(image)

    affine_corrected=affine_correct(image,points)
    metric_corrected=metric_correction(image,points)
    mosaic_image=np.concatenate((image,affine_corrected,metric_corrected),axis=1)
    cv2.imwrite("AffineCorrected.jpg",affine_corrected)
    cv2.imwrite("MetricCorrected.jpg",metric_corrected)
    cv2.imwrite("Mosaiced.jpg",mosaic_image)


if __name__=="__main__":
    main()