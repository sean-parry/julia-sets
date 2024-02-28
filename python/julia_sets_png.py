import numpy as np ,os, cv2

'''
this generates 'pgm' images so its just black and white i should add colour to the image and get it plotted with matplot lib and have sliders
instead of inputting the values in main
'''
def generate_julia_set(re_val, im_val):
    """
    generates a julia set
    """
    width, height = 1080, 1080

    re_min, re_max, im_min, im_max = -2.0, 2.0, -2.0, 2.0
    file_name = 'output.png'
    c = complex(re_val, im_val)
    real_range = np.arange(re_min, re_max,(re_max-re_min)/width)
    im_range = np.arange(im_min, im_max,(im_max-im_min)/height)

    img = np.zeros((height, width, 3), dtype=np.uint8)

    if os.path.exists(file_name):
        os.remove(file_name)


    for j,im in enumerate(im_range):
        for i,re in enumerate(real_range):
            z = complex(re,im)
            n = (0,0,255)
            while abs(z) < 10 and n[0]<255: 
                z = z*z+c
                if n[1] >= 255:
                    n  = (n[0]+15,n[1],n[2])
                else:
                     n  = (n[0],n[1]+15,n[2])
            img[i,j] = n

    cv2.imwrite(file_name,img)
    print('created julia set')
    
def main():
    generate_julia_set(0.0, 0.78)

if __name__ == '__main__':
    main()

"""
For png start 0,0,255
"""