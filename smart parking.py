import cv2
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import numpy as np
from pytesseract import image_to_string
from copy import deepcopy
import time
import os
import RPi.GPIO as GPIO
import picamera
import MySQLdb
import Adafruit_CharLCD as LCD

##############################################
# Subplot generator for images
def plot(figure, subplot, image, title, cmap=None):
    figure.subplot(subplot)
    figure.imshow(image, cmap)
    figure.xlabel(title)
    figure.xticks([])
    figure.yticks([])


def satisfy_ratio(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    error = 0.4
    aspect = 4.1667 # In India, car plate size: 500x120 aspect 4.1667
    # Set a min and max area. All other patches are discarded
    min = 15*aspect*15  # minimum area
    max = 125*aspect*125  # maximum area
    # Get only patches that match to a respect ratio.
    rmin = aspect - aspect*error
    rmax = aspect + aspect*error

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True


def verify_sizes(rectangle):

    (x, y), (width, height), rect_angle = rectangle

    # Calculate angle and discard rects that has been rotated more than 15 degrees
    angle = 90 - rect_angle if (width < height) else -rect_angle
    if 15 < abs(angle) < 165:  # 180 degrees is maximum
        return False
    # We only consider that a region can be a plate if the aspect ratio is approximately 4.1667
    # (plate width divided by plate height) with an error margin of 40 percent
    # and an area based on a minimum of 15 pixels and maximum of 125 pixels for the height of the plate.
    # These values are calculated depending on the image sizes and camera position:
    area = height * width

    if height == 0 or width == 0:
        return False
    if not satisfy_ratio(area, width, height):
        return False

    return True


def get_dominant_color(plate):
    average_color = np.mean(plate).astype(np.uint8)
    return average_color


def is_white_color_dominant(plate):
    average_color = np.mean(plate).astype(np.uint8)
    return 100 <= average_color  # white color is dominant if mean color > 100


def plot_plate_numbers(plates_images):
    ''' Plot Plate Numbers as separate images '''
    i = 0
    for plate_img in plates_images:
        cv2.imshow('plate-%s' % i, plate_img)
        cv2.resizeWindow("plate-%s" % i, 300, 40)
        cv2.imwrite('plates/plate-%s.jpg' % i, plate_img)
        i += 1


def preprocess_plate(plate_img):
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)  # make greyscale

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    dilate_thresh = cv2.dilate(plate_gray, kernel, iterations=1)  # dilate
    _, thresh = cv2.threshold(dilate_thresh, 150, 255, cv2.THRESH_BINARY)  # threshold

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

    if contours:
        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        max_cnt = contours[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        # Can't satisfy ratio, exit
        max_cnt_area = areas[max_index]
        if not satisfy_ratio(max_cnt_area, w, h):
            return plate_img, None

        final_plate_img = plate_img[y:y + h, x:x + w]  # crop and fetch only plate number

        # # for each contour found, draw a rectangle around it on original image
        plate_img_with_contours = plate_img.copy()
        cv2.drawContours(plate_img_with_contours, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)

        return final_plate_img, [x, y, w, h]
    else:
        return plate_img, None


def find_contours(img):
    '''
    :param img: (numpy array)
    :return: all possible rectangles (contours)
    '''
    img_blurred = cv2.GaussianBlur(img, (5, 5), 1)  # remove noise
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)  # greyscale image
    

    # Apply Sobel filter to find the vertical edges
    # Find vertical lines. Car plates have high density of vertical lines
    img_sobel_x = cv2.Sobel(img_gray, cv2.CV_8UC1, dx=1, dy=0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    

    # Apply optimal threshold by using Oslu algorithm
    retval, img_threshold = cv2.threshold(img_sobel_x, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)


    # Define a stuctural element as rectangular of size 17x3 (we'll use it during the morphological cleaning)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))

    # And use this structural element in a close morphological operation
    morph_img_threshold = deepcopy(img_threshold)
    cv2.morphologyEx(src=img_threshold, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img_threshold)


    # Find contours that contain possible plates (in hierarchical relationship)
    contours, hierarchy = cv2.findContours(morph_img_threshold,
                                           mode=cv2.RETR_EXTERNAL,  # retrieve the external contours
                                           method=cv2.CHAIN_APPROX_NONE)  # all pixels of each contour

    plot_intermediate_steps = False
    if plot_intermediate_steps:
        plot(plt, 321, img, "Original image")
        plot(plt, 322, img_blurred, "Blurred image")
        plot(plt, 323, img_gray, "Grayscale image", cmap='gray')
        plot(plt, 324, img_sobel_x, "Sobel")
        plot(plt, 325, img_threshold, "Threshold image")
        plt.tight_layout()
        plt.show()

    return contours


def find_plate_numbers(origin_img, contours, mask):
    # For each contour detected, extract the bounding rectangle of minimal area and validate every contour
    # before classifying every region

    plot_all_found_contours = False
    plates, plates_images = [], []
    time_start = time.time()
    for rect_n, cnt in enumerate(contours):
        min_rectangle = cv2.minAreaRect(cnt)
        # Debug: keep track of found contours
        if plot_all_found_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(origin_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(origin_img, str(rect_n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 255, 255))

        if verify_sizes(min_rectangle):
            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = deepcopy(origin_img[y:y + h, x:x + w])  # crop
            clean_plate_img, plate_rect = preprocess_plate(plate_img)

            # Rebuild coords for cleaned plate (if the plate number has been cleared more precisely)
            if plate_rect:
                x1, y1, w1, h1 = plate_rect
                x, y, w, h = x + x1, y + y1, w1, h1

            # In order to make it faster, filter by dominant color on the 2nd step
            if is_white_color_dominant(clean_plate_img):
                # Try to parse vehicle number

                # Apply Tesseract app to parse plate number
                plate_im = Image.fromarray(clean_plate_img)


                t_start = time.time()
                plate_text = image_to_string(plate_im, lang='eng')
                plate_text = plate_text.replace(' ', '').upper()
                print 'Time taken to extract text from contour: %s' % (time.time() - t_start)

                # TODO: Use faster method to detect (Tesseract requires ~0.5 sec to identify text)
               
                    # Draw rectangle around plate number
                cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
                plates.append(plate_text)
                plates_images.append(clean_plate_img)

    print "Time taken to process contours: %s" % (time.time() - time_start)
    # Debug: Plot all found contours
    if plot_all_found_contours:
        cv2.imshow('', origin_img)
        cv2.waitKey(0)
        exit()
    return plates, plates_images, mask

###########################################
# Main methods ############################
###########################################
def process_single_image(images=[], plot_plates=False):
    '''
    :param images: list (full path to images to be processed)
    '''
    if images:
        img_n = 1
        for path_to_image in images:
            t_start = time.time()
            img = cv2.imread(path_to_image)

            # Resizing of the image
            r = 350.0 / img.shape[1]
            dim = (350, int(img.shape[0] * r))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            mask = np.zeros_like(img)  # init mask

            contours = find_contours(img)

            plates, plates_images, mask = find_plate_numbers(img, contours, mask)

            print "Time needed to complete: %s" % (time.time() - t_start)
            print "Plate Numbers: %s" % plates
            #print "Plate Numbers: %s" % ", ".join(plates)
            numplate=plates
            print numplate;
            # Apply mask to image and plot image
            img = cv2.add(img, mask)

            if plot_plates:
                plot_plate_numbers(plates_images)

            cv2.imshow('Resized Original image_%s + Detected Plate Number' % img_n, img)
            img_n += 1

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return numplate
    else:
        exit('Images are not provided!')




def start():
    test_plates = [os.path.join('test_images', im)  for im in os.listdir("test_images")]
    images = ['test_images/1o.jpg']

    x=process_single_image(images)
    return x

###########################################
# Run The Program #########################
###########################################

if __name__ == '__main__':
    trig1=23
    echo1=24
    trig2=17
    echo2=27
    ldr=22
    led=4
    green1=6
    red1=13
    buzzer1=5
    lcd_rs = 25
    lcd_en = 8
    lcd_d4 = 7
    lcd_d5 = 12
    lcd_d6 = 16
    lcd_d7 = 20
    lcd_backlight = 21

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(trig1,GPIO.OUT)
    GPIO.setup(echo1,GPIO.IN)
    GPIO.setup(trig2,GPIO.OUT)
    GPIO.setup(echo2,GPIO.IN)
    GPIO.setup(ldr,GPIO.IN)
    GPIO.setup(led,GPIO.OUT)
    GPIO.setup(red1,GPIO.OUT)
    GPIO.setup(green1,GPIO.OUT)
    GPIO.setup(buzzer1,GPIO.OUT)
    GPIO.setup(lcd_rs,GPIO.OUT)
    
    camera = picamera.PiCamera()
    lcd_columns = 16
    lcd_rows = 2

    lcd = LCD.Adafruit_CharLCD(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows, lcd_backlight)

    s='/home/pi/project/test_images/1o.jpg'

    db=MySQLdb.connect('localhost','root','mediawiki','smart_parking')
    curs=db.cursor()

    j=0
    while j<1:
        #turnng off any previous trigger
        GPIO.output(trig1,0)
        GPIO.output(trig2,0)
        GPIO.output(trig3,0)
        GPIO.output(trig4,0)
        time.sleep(1)

        #sending pulse from u1
        GPIO.output(trig1,1)
        time.sleep(0.00001)
        GPIO.output(trig1,0)

        i=0
        while GPIO.input(echo1)==0 and i<350:
            st1=time.time()
            i+=1

        if GPIO.input(echo1)==1:
            while GPIO.input(echo1)==1:
                et1=time.time()
        
            #sending pulse from u2
            GPIO.output(trig2,1)
            time.sleep(0.00001)
            GPIO.output(trig2,0)

            while GPIO.input(echo2)==0:
                st2=time.time()
            while GPIO.input(echo2)==1:
                et2=time.time()

            #distance measurement
            t1=et1-st1
            t2=et2-st2
            d1=t1*17150
            d2=t2*17150

            DISTANCE1=d1-0.5
            DISTANCE2=d2-0.5

            #checking for slot 1
            if d1>20 and d1<200 and d2>20 and d2<200 :
                print 'Distance1 is ',DISTANCE1,' cm'
                print 'Distance2 is ',DISTANCE2,' cm'

                #checking for light
                if GPIO.input(ldr)==0:
                    GPIO.output(led,1)
                else:
                    GPIO.output(led,0)
            
                #click picture
                camera.start_preview()
                time.sleep(5)
                camera.capture(s)
                camera.close()
                numplate=start()
                n=str(numplate)
                print n;
                r=None
                curs.execute("SELECT * FROM details WHERE carnum='"+n+"'")
                result=curs.fetchall()

                for r in result:
                    print r
                if r!=None:
                    print 'number plate found. Car can be parked.'
                    GPIO.output(green1,1)
                    lcd.message('Car can be parked')
                    time.sleep(2)
                    lcd.clear()
                else:
                    print 'number plate not found'
                    GPIO.output(red1,1)
                    GPIO.output(buzzer1,1)
                    lcd.message('Car cannot be parked')
                    time.sleep(2)
                    lcd.clear()
                
                curs.close()
                db.close()
                
                GPIO.output(green1,0)
                GPIO.output(red1,0)
                GPIO.output(buzzer1,0)

                #closing light if turned on for photo
                if GPIO.OUT(led):
                    GPIO.output(led,0)
            
            else:
                print 'out of range'

        j+=1
