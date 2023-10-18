import cv2                                                                                          
import os
import time
from screeninfo import get_monitors
import numpy as np          

def get_screen_resolutions():
    monitors = get_monitors()
    resolutions = [(monitor.width, monitor.height) for monitor in monitors]
    return resolutions     
    
def create_black_background(height, width):
    return np.zeros((height, width, 3), dtype="uint8")
                                                           
                                                                                                    
def get_newest_images(folder_path, num_images):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        return []

    def sort_key_func(file_name):
        try:
            return int(os.path.splitext(file_name.split('_')[1])[0])
        except ValueError:
            return float('-inf')

    files.sort(key=sort_key_func, reverse=True)
    newest_files = files[:num_images]
    images = [cv2.imread(os.path.join(folder_path, file)) for file in newest_files]
    images = [img for img in images if img is not None]
    return images

                                                                              
                                                                                                    
def display_images(images, window_name='CougarVision'):                                            
    h, w, _ = images[0].shape  
    # Create a black image with the necessary size
    display_img = np.zeros((h*3, w*3, 3), np.uint8)                                 
    for i in range(len(images)):                                                                    
        x_offset = (i % 3) * w                                                                      
        y_offset = (i // 3) * h                                                                     
        display_img[y_offset:y_offset + h, x_offset:x_offset + w] = images[i]                       
    cv2.imshow(window_name, display_img) 

def display_more_images(images, window_name='Newest Image'):                                          
    # Check if there are any images to display
    if not images:
        print("No images to display")
        return

    h, w, _ = images[0].shape

    display_img = create_black_background(h*9, w*9)

    for i in range(len(images)):
        if images[i] is not None:   
            x_offset = (i % 9) * w                                                                      
            y_offset = (i // 9) * h
            
            resized_image = cv2.resize(images[i], (w, h))
            display_img[y_offset:y_offset + h, x_offset:x_offset + w] = resized_image                       
    
    cv2.imshow(window_name, display_img)


                                                          
                                                                                                    
if __name__ == "__main__":                                                                          
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
        folder_path_for_9 = '/home/katiedemo/demo_images'  # your existing directory
        num_images = 9
        newest_images = get_newest_images(folder_path_for_9, num_images)
        if len(newest_images) >= 9:
            display_images(newest_images, window_name)

        # For the 81-image display
        folder_path_for_81 = '/home/katiedemo/unlabeled_photos'  # your new directory for 81 images
        number_images = 81
        newester_images = get_newest_images(folder_path_for_81, number_images)                                              
        if len(newester_images) >= 81:
            display_more_images(newester_images, newest_window_name)                                            
                                                                                                    
        time.sleep(1)                                                                               
                                                                                                    
        if cv2.waitKey(1) & 0xFF == ord('q'):                                                       
            break                                                                                   
                                                                                                    
    cv2.destroyAllWindows()                                                                         
                            

