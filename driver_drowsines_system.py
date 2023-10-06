from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlibd
import cv2
from twilio.rest import Client
import os


account_sid = 'AC56b4d0a664eb00b21dc3ade0599b2ce1'
auth_token = '2f7ba62c35c60e4dcd391e6f9fe7a2c3'
client = Client(account_sid, auth_token)



 
mixer.init()
mixer.music.load(r'D:\DL_projrcts\project_opencv\music.wav')

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r'D:\DL_projrcts\project_opencv\shape_predictor_68_face_landmarks (1).dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
video=cv2.VideoCapture('D:\DL_projrcts\project_opencv\Driver drowsiness detection - video num 1.mp4')
flag=0

while True:
	suc, frame=video.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)


		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		 
		if ear < thresh:
			flag += 1
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "***********Drowsiness Detected****************", (10,250),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				mixer.music.play()
				call = client.calls.create(
                        twiml='hlo sir',
                        to='+16189613413',
                        from_='+916282407302'
                    )
				
             
			     
					     
				
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("d"):
		break
print(call.sid)
	
cv2.destroyAllWindows()
video.release() 