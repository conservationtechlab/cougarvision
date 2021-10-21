import cv2

cap = cv2.VideoCapture('rtsp://admin:NyalaChow22@localhost:8080')
frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
writer = cv2.VideoWriter(f'testing_rtsp.avi', 
                        cv2.VideoWriter_fourcc(*'XVID'),
                        20, frameSize)
while(cap.isOpened()):
    ret,frame= cap.read()
    if ret == True:
        print('Successful read')
        writer.write(frame)
    else:
        print('False read')

        

