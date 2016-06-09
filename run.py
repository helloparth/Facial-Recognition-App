import cv2
import sys
import scipy.ndimage as ndimage



cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

user = ndimage.imread('/Users/ppatel/Projects/FaceRecognition/parth.jpg')

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def compare_images(img1, img2):

    hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
    return cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)

  	





#take input from webcam
video_capture = cv2.VideoCapture(0)
i = 0
while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30,30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	)

	cropped = None
	largestArea = 0
	trueX,trueY,trueW,trueH = 0,0,0,0
	for (x, y, w, h) in faces:
		if w*h > largestArea:
			largestArea = w*h
			trueX,trueY,trueW,trueH = x,y,w,h

	if largestArea != 0:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (200,250,100), 10)
		font = cv2.FONT_HERSHEY_SIMPLEX
		#cv2.putText(frame, "Hi Parth!",(10,20),font, 1.0, 250)
		cropped = frame[y:y+h, x:x+h]

	key = cv2.waitKey(1)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	if i % 1 == 0:
		score =  compare_images(user,cropped)
		font = cv2.FONT_HERSHEY_SIMPLEX
		if score < 0.09:
			print 'match: ', score
			cv2.putText(frame, "Detecting.. Parth",(50,50),font, 1.2, (200, 255, 0), 3)
		elif score < 0.7:
			print 'mismatch: ', score
			cv2.putText(frame, "Detecting.. Some Face",(50,50),font, 1.2, (0, 255, 240), 3)
		else:
			print 'nothing detected: ', score
			cv2.putText(frame, "No face detected",(50,50),font, 1.2, (20, 0, 250), 3)


	cv2.imshow('Video', frame)

	i+=1

	


video_capture.release()
cv2.destroyAllWindows()
