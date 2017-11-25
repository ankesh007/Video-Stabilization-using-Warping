import numpy as np
import cv2

# img1 = cv2.imread()
# img2 = cv2.imread()
			# img1 = np.arange(24).reshape([6,4])
			# shape = np.shape(img1)

			# p = np.zeros([shape[0], shape[1], 6])
			# p[:,:,0] = 1
			# p[:,:,3] = 1
			# p[:,:,4] = 1

			# def warp(img):
			# 	t_img = np.zeros(shape)
			# 	x_series = np.arange(shape[0])
			# 	y_series = np.arange(shape[1])
			# 	indice = np.indices(shape)
			# 	# # print np.shape(indice)
			# 	# print np.vstack([indice, np.expand_dims(np.zeros(shape), axis=0)])
			# 	# v = np.vstack([indice, np.expand_dims(np.zeros(shape), axis=0)]).astype('int')
			# 	# print np.shape(p)
			# 	# # print v
			# 	# print np.indices([6, 4, 6])
			# 	# print np.shape(np.indices([6,4,6]))
			# 	# print np.shape(p[np.indices([6, 4, 6])])
			# 	# print p
			# 	# print p[indice[0], indice[1], 2]
			# 	# print p[indice[0], indice[1], 3]
			# 	# # print np.shape(p)
			# 	# print np.shape(p[indice[0], indice[1], 4])
			# 	xs = (p[indice[0], indice[1], 0]*indice[0] + p[indice[0], indice[1], 1]*indice[1] + p[indice[0], indice[1], 4]).astype('int').clip(0,5)
			# 	ys = (p[indice[0], indice[1], 2]*indice[0] + p[indice[0], indice[1], 3]*indice[1] + p[indice[0], indice[1], 5]).astype('int').clip(0,5)
			# 	rang = np.vstack([np.expand_dims(xs, axis=0), np.expand_dims(ys, axis=0)])
			# 	t_img[indice[0], indice[1]] = img[rang[0], rang[1]]
			# 	return t_img

			# warped_img = warp(img1)
			# print img1
			# print warped_img


template = cv2.imread('template.JPG')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
target_img = cv2.imread('target.JPG')
target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
template = template.astype('float32')
target_img = target_img.astype('float32')
print "Read"


print np.shape(template)
print np.shape(target_img)
# template = np.arange(5000).reshape([100, 50]).astype('float')
dtdx = cv2.Sobel(template, cv2.CV_32F, 1, 0)
dtdy = cv2.Sobel(template, cv2.CV_32F, 0, 1)
t_h = np.shape(template)[0]
t_w = np.shape(template)[1]
coord = np.indices([t_h, t_w])
# print coord
# print np.shape(coord)
temp = np.copy(coord[0,:,:])
coord[0,:,:] = coord[1,:,:]
coord[1,:,:] = temp
coord  = np.concatenate([coord,np.expand_dims(np.ones([t_h, t_w]), axis =0)], axis=0)
coord = np.moveaxis(coord, 0, -1)
# print coord
temp_matr = np.array([[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1],[0,0,0]])
temp_matr1 = np.array([[0,0,0],[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1]])
temp_matr = np.broadcast_to(temp_matr, [t_h, t_w, 6, 3])
temp_matr1 = np.broadcast_to(temp_matr1, [t_h, t_w, 6, 3])
dwdp_0 = np.matmul(temp_matr,np.expand_dims(coord, axis=3))
dwdp_1 = np.matmul(temp_matr1,np.expand_dims(coord, axis=3))
dwdp = np.concatenate([dwdp_1,dwdp_0], axis = 3)
dwdp = np.transpose(dwdp, (0,1,3,2))
# print dwdp
delt = np.stack([dtdy, dtdx], axis=2)
delt = np.expand_dims(delt, axis=2)
# print np.shape(delt)
sdi = np.matmul(delt, dwdp)
# print dwdp
print sdi
print sdi.shape
# print np.shape(sdi)
hessian = np.matmul(np.transpose(sdi, (0,1,3,2)), sdi)
hessian = np.sum(hessian, axis = 0)
hessian = np.sum(hessian, axis=0)
hessian_inv = np.linalg.inv(hessian)
# print hessian_inv
d = np.array([1,0,0,1,0,0]).astype('float32')
i=1
warpt = np.ones(np.shape(template))
update = np.ones([6, 1])
# print template.shape
# exit(1)
print "Entering"
i=0
# print np.linalg.norm(update, axis = 0)
while np.linalg.norm(update, axis = 0) > 0.001:
	warpt = cv2.warpAffine(target_img, np.transpose(np.reshape(d,[3,2])), (template.shape[1],template.shape[0]))
	np.savetxt("lost2.csv",warpt,delimiter=',')
	error = warpt - template
	expr = np.sum(np.sum(np.matmul(np.transpose(sdi, (0,1,3,2)), np.expand_dims(np.expand_dims(error, axis=-1),axis=-1)), axis=0), axis=0)
	# print np.matmul(np.transpose(sdi, (0,1,3,2)), np.expand_dims(np.expand_dims(error, axis=-1),axis=-1))
	# print expr
	print np.expand_dims(np.expand_dims(error, axis=-1),axis=-1).shape
	exit(1)
	update = np.matmul(hessian_inv, expr)
	print update
	
	dnew = np.ones([6,1])

	dtemp = d
	# print d
	dnew[0] = dtemp[0] + update[0] + dtemp[0]*update[0] + dtemp[2]*update[1]
	dnew[1] = dtemp[1] + update[1] + dtemp[1]*update[0] + dtemp[3]*update[1]
	dnew[2] = dtemp[2] + update[2] + dtemp[0]*update[2] + dtemp[2]*update[3]
	dnew[3] = dtemp[3] + update[3] + dtemp[1]*update[2] + dtemp[3]*update[3]
	dnew[4] = dtemp[4] + update[4] + dtemp[0]*update[4] + dtemp[2]*update[5]
	dnew[5] = dtemp[5] + update[5] + dtemp[1]*update[4] + dtemp[3]*update[5]
	print str(i) + " " + str(np.linalg.norm(update, axis = 0))
	d = dnew
	i = i + 1
final_img = cv2.warpAffine(target_img, np.transpose(np.reshape(d,[3,2])), (template.shape[1],template.shape[0]))
r = np.array([[9.96681990e-01,3.83096730e-03,4.60417403e+01],[3.16118120e-03,9.97444464e-01, 8.41931344e-01]])
aman = cv2.warpAffine(target_img, r, (template.shape[1],template.shape[0]))
# cv2.imshow('image',final_img)
cv2.imshow('image', final_img.astype('uint8'))
cv2.imshow('aman', aman.astype('uint8'))

cv2.waitKey(0)
cv2.destroyAllWindows()










