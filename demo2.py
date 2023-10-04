import cv2                                                                                          
import os
import time
from screeninfo import get_monitors
import numpy as np          

def get_screen_resolutions():
    monitors = get_monitors()
    resolutions = [(monitor.width, monitor.height) for monitor in monitors]
    return resolutions                                                                      
                                                                                                    
def get_newest_images(folder_path, num_images):                                                                 
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]    
    files.sort(key=lambda x: int(os.path.splitext(x.split('_')[1])[0]))                             
    newest_files = files[-num_images:]                                                                       
    images = [cv2.imread(os.path.join(folder_path, file)) for file in newest_files]                 
    return images                                                                                 
                                                                                                    
def display_images(images, window_name='CougarVision'):                                            
    h, w, _ = images[0].shape                                                                       
    display_img = cv2.resize(cv2.imread('black.jpg'), (w*3, h*3))                                   
    for i in range(len(images)):                                                                    
        x_offset = (i % 3) * w                                                                      
        y_offset = (i // 3) * h                                                                     
        display_img[y_offset:y_offset + h, x_offset:x_offset + w] = images[i]                       
    cv2.imshow(window_name, display_img) 
    
def display_more_images(images, window_name='Newest Image'):                                          
    h, w, _ = images[0].shape                                                                       
    display_img = cv2.resize(cv2.imread('black.jpg'), (w*9, h*9))                                   
    for i in range(len(images)):                                                                    
        x_offset = (i % 9) * w                                                                      
        y_offset = (i // 9) * h                                                                     
        display_img[y_offset:y_offset + h, x_offset:x_offset + w] = images[i]                       
    cv2.imshow(window_name, display_img)                                                           
                                                                                                    
if __name__ == "__main__":                                                                          
    folder_path = '/home/katie/Documents/cougarvision/demo_images'  # Change this to your folder 
    window_name = 'CougarVision'                                                                    
    newest_window_name = "Newest Image"                                                              
    
    resolutions = get_screen_resolutions() 
    
    image1 = np.zeros((resolutions[0][1], resolutions[0][0], 3), np.uint8)
    image2 = np.zeros((resolutions[0][1], resolutions[0][0], 3), np.uint8) 

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(newest_window_name, cv2.WINDOW_NORMAL)

    cv2.moveWindow(window_name, 0, 0)
    cv2.moveWindow(newest_window_name, resolutions[0][0], 0)

    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(newest_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)     
                                                                                                    
    while True:     
        num_images = 9                                                                                
        newest_images = get_newest_images(folder_path, num_images)
        if len(newest_images) >= 9:
            display_images(newest_images, window_name)
        number_images = 81
        newester_images = get_newest_images(folder_path, number_images)                                              
        if len(newester_images) >= 81:
            display_more_images(newester_images, newest_window_name)                                              
                                                                                                    
        time.sleep(1)                                                                               
                                                                                                    
        if cv2.waitKey(1) & 0xFF == ord('q'):                                                       
            break                                                                                   
                                                                                                    
    cv2.destroyAllWindows()                                                                         
                            

