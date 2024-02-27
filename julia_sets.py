import numpy,os
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
    file_name = 'output.pgm'
    c = complex(re_val, im_val)
    real_range = numpy.arange(re_min, re_max,(re_max-re_min)/width)
    im_range = numpy.arange(im_min, im_max,(im_max-im_min)/height)

    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name,'w') as f:
        f.write(f'P2\n# julia set image\n{str(width)} {str(height)}\n255\n') # header info for file

        for im in im_range:
            for re in real_range:
                z = complex(re,im)
                n = 255
                while abs(z) < 10 and n>=5:
                    z = z*z+c
                    n-=5
                f.write(f'{str(n)} \n')
    print('created julia set')
    
def main():
    generate_julia_set(0.0, 0.78)

if __name__ == '__main__':
    main()