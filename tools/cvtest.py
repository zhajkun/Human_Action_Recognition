
import sys
from random import randint
import cv2 as cv
def main(argv):
    
    borderType = cv.BORDER_CONSTANT
    window_name = "copyMakeBorder Demo"
    
    imageName = argv[0] if len(argv) > 0 else 'lena.jpg'
    # Loads an image
    src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: copy_make_border.py [image_name -- default lena.jpg] \n')
        return -1
    
    print ('\n'
           '\t   copyMakeBorder Demo: \n'
           '     -------------------- \n'
           ' ** Press \'c\' to set the border to a random constant value \n'
           ' ** Press \'r\' to set the border to be replicated \n'
           ' ** Press \'ESC\' to exit the program ')
    
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    
    top = int(0.0 * src.shape[0])
    bottom = top
    right = int(0.0 * src.shape[1])
    left = int(0.3 * src.shape[1])  # shape[1] = cols

    
    while 1:
        
        value = [255, 255, 255]
        
        dst = cv.copyMakeBorder(src, top, bottom, left, right, borderType, None, value)
        
        cv.imshow(window_name, dst)
        
        c = cv.waitKey(500)
        if c == 27:
            break
        elif c == 99: # 99 = ord('c')
            borderType = cv.BORDER_CONSTANT
        elif c == 114: # 114 = ord('r')
            borderType = cv.BORDER_REPLICATE
        
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])