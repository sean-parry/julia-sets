import numpy as np , matplotlib.pyplot as plt, threading, time
"""
do some prints upon initialisation of julia set
make sure all threads are terminated before starting or stopping
and restart ur pc
when i enter a class is the main stack dropped or should i just be using the julia set obj without sending it into the function?
"""
class JuliaSet():
    def __init__(self,width, height, c, max_iter):
        k, i = np.ogrid[1.4: -1.4: height*1j, -1.4: 1.4: width*1j]
        self.z_array = i + k*1j
        self.c = c
        self.max_iter = max_iter
        self.iter_until_diverge = max_iter + np.zeros((width,height))
        self.print_julia_set()
        print(self.iter_until_diverge.shape)
        self.lock = threading.Lock()

    def get_z(self,x,y):
        return self.z_array[x,y]
    
    def add_diverged_val(self,x,y,diverged_iter):
        with self.lock:
            self.iter_until_diverge[x,y] = diverged_iter
    
    def print_julia_set(self):
        print(self.iter_until_diverge)


class PixelThread(threading.Thread):
    def __init__(self,x,y,julia_obj):
        super().__init__()
        self.julia_obj = julia_obj
        self.x = x
        self.y = y
        self.curr_iter = -1
        self.z = julia_obj.get_z(x,y)
        self._stop_event = threading.Event()

    def run(self):
        for i in range(self.julia_obj.max_iter):
            self.curr_iter +=1
            self.z = self.z**2 + self.julia_obj.c
            if self.z *np.conj(self.z) > 4:
                self.julia_obj.add_diverged_val(self.x,self.y,self.curr_iter)
                break
        # with PixelThread.lock:  # Acquire the lock before printing
        #     print(f'thread terminated with iter val of: {self.curr_iter}')
        self._stop_event.set()
      

def main():
    width, height,c, max_iter = 1000,1000,-0.744+0.148*1j,10
    julia_obj = JuliaSet(width,height,c,max_iter)
    threads = []
    for j in range (height):
        # print(f'{j}')
        for i in range (width):
            thread = PixelThread(i,j,julia_obj)
            thread.start()
            threads.append(thread)
    print('joining threads')
    # waits for all thread to complete
    for thread in threads:
        thread.join()

    print('done')

    plt.imshow(julia_obj.iter_until_diverge, cmap='twilight_shifted')
    plt.axis('off')
    plt.show()
    

if __name__ == '__main__':
    main()