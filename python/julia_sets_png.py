import numpy as np , cv2

def generate_julia_set(c, width =1080, height=1080,re_min=-2.0,re_max=2.0,im_min=-2.0,im_max=2.0,file_name='output.png'):
    """
    generates a julia set
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    for j in range(height):
        im = im_min +(im_max-im_min)*j/height
        for i in range(width):
            re = re_min +(re_max-re_min)*i/width
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
    generate_julia_set(complex(0.0, 0.78))

if __name__ == '__main__':
    main()