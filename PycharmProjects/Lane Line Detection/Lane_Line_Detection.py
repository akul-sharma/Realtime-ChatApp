# Steps for lane line detectin:
# step1: Read and Decode file into frames
# step2: Greyscale coversion of image
# step3: Reduce noise by applying filter
# step4: Detecting the edges
# step5: Mask the canny image
# step6: Find the coordinates of road lane
# step7: Fit the coordinates into canny image
# step8: Edge detection is done

import cv2
import numpy as np

# This Function will select the area that is of our region of interest
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# This function will draw the lines on blank image that has the same coordinates as that of image
# And finally it will merge the two images so that lines can be seen on original frame
def drow_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:     # where (x1,y1) and (x2,y2) are coordinates of the two points of line
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    # Merging the two images
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def process(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    #Selecting vertices according to our need
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/1.3),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    #Canny Edge Detection
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = drow_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('test.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()