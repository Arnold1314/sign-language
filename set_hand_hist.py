import cv2
import numpy as np
import pickle

def build_squares(img):#建立绿色矩形方阵
	x, y, w, h = 420, 140, 10, 10
	d = 10
	imgCrop = None
	crop = None
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				imgCrop = img[y:y+h, x:x+w]
			else:
				imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))#沿着水平方向将数组堆叠起来。
			#print(imgCrop.shape)
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)#绘制矩形
			x+=w+d
		if np.any(crop == None):
			crop = imgCrop
		else:
			crop = np.vstack((crop, imgCrop)) #按垂直方向（行顺序）堆叠数组构成一个新的数组
		imgCrop = None
		x = 420
		y+=h+d
	return crop

def get_hand_hist():#获取手势直方图
	cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)#打开笔记本摄像头
	if cam.read()[0]==False:#按帧读取视频
		cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	x, y, w, h = 300, 100, 300, 300
	flagPressedC, flagPressedS = False, False
	imgCrop = None
	while True:
		img = cam.read()[1]#按帧读取视频
		img = cv2.flip(img, 1)#水平翻转图像
		img = cv2.resize(img, (640, 480))#修改图片大小
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#BGR和HSV图像的转换，RGB颜色空间更加面向于工业，而HSV更加面向于用户，大多数做图像识别这一块的都会运用HSV颜色空间，因为HSV颜色空间表达起来更加直观！
		
		keypress = cv2.waitKey(1)#返回一个按下的键
		if keypress == ord('c'):#返回对应的ASCII值
			hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)#BGR和HSV图像的转换
			flagPressedC = True#按下C的标志为true
			hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])#计算图像直方图

			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)#图像归一化
		elif keypress == ord('s'):#如果按键为s则令flagPressedS为True
			flagPressedS = True	
			break
		if flagPressedC:	
			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
			'''对照直方图进行反向投影，反向投影用于在输入图像（通常较大）
			中查找与模板图像（通常较小甚至仅 1 个像素）最匹配的点或区域，也就是确定模板图像在输入图像中的位置。'''
			dst1 = dst.copy()
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))#内核形状为椭圆形，第二和第三个参数分别是内核的尺寸以及锚点的位置
			cv2.filter2D(dst,-1,disc,dst)#图像卷积运算函数
			blur = cv2.GaussianBlur(dst, (11,11), 0)#高斯滤波，使得图像平滑
			blur = cv2.medianBlur(blur, 15)#中值滤波，非线性平滑技术，它将每一像素点的灰度值设置为该点某邻域窗口内的所有像素点灰度值的中值
			ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#图像二值化，就是图中黑白效果的来源
			thresh = cv2.merge((thresh,thresh,thresh))#通道合并函数
			#cv2.imshow("res", res)
			cv2.imshow("Thresh", thresh)#在窗口显示图像Tresh
		if not flagPressedS:
			imgCrop = build_squares(img)
		#cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.imshow("Set hand histogram", img)#显示摄像头读取的图像
	cam.release()#释放摄像头
	cv2.destroyAllWindows()#关闭窗口并取消内存分配
	with open("hist", "wb") as f:
		pickle.dump(hist, f)#将hist保存到f中去


get_hand_hist()
