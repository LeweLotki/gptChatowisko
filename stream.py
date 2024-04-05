import cv2
import os
from shutil import rmtree
from config import paths
import numpy as np

class Stream:

    frame_count = 0
    output_dir = paths.stream_frames_path
    left_dir, right_dir = (None, None)

    def __init__(self):

        self.cap = cv2.VideoCapture(0)

    def run(self, mode:str='display_mode', output_dir=None, frame_limit=int(1e3)):
        
        if output_dir is None:
            output_dir = self.output_dir

        if mode == 'display_mode':
            self.__display_mode()

        elif mode == 'save_display_mode':
            self.__save_display_mode(frame_limit=frame_limit)

        elif mode == 'save_mode':
            self.__save_mode(frame_limit=frame_limit)

        elif mode == 'void_mode':
            self.__void_mode(frame_limit=frame_limit)
            
        else:

            print('Fatal error: Incorrect argument mode, does not match any defined modes.')

    def get_single_frame(self, recursion_depth=0) -> (np.array, np.array):

        ret, frame = self.cap.read()
        max_recursion_depth = 10
        
        if not ret: return (None, None)
        if recursion_depth > max_recursion_depth: return (None, None)
        if (np.min(frame[0,:,0]) == np.max(frame[0,:,0])):

           recursion_depth += 1
           (left_image, right_image) = self.get_single_frame(recursion_depth=recursion_depth) 
           return (left_image, right_image)

        (
            left_image,
            right_image
        ) = self.__subdivide_camera_image(frame)                

        return (left_image, right_image)
       
    def __save_display_mode(self, frame_limit=None):

        self.__create_output_dir()
        try:
            while (frame_limit == None) or frame_limit >= self.frame_count:
                ret, frame = self.cap.read()
                if not ret:
                    break 
                else:
                    self.frame_count += 1

                self.__display_images(frame)
                self.__save_images(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def __display_mode(self):
        
        frame_saved = False

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break 
                else:
                    self.frame_count += 1

                self.__display_images(frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not frame_saved:
                        self.__create_output_dir()
                        frame_saved = True
                    self.__save_images(frame)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def __save_mode(self, frame_limit:int=int(1e3)):

        self.__create_output_dir()
        try:
            while frame_limit >= self.frame_count:
                ret, frame = self.cap.read()
                if not ret:
                    break 
                else:
                    self.frame_count += 1

                self.__save_images(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def __void_mode(self, frame_limit:int=int(1e3)):
        
        frame_saved = False
        
        try:
            while frame_limit >= self.frame_count:
                ret, frame = self.cap.read()
                if not ret:
                    break 
                else:
                    self.frame_count += 1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if not frame_saved:
                        self.__create_output_dir()
                        frame_saved = True
                    self.__save_images(frame)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def __create_output_dir(self, path=None):
      
        if path is None:
            path = self.output_dir

        if os.path.exists(path):
            rmtree(path)

        self.left_dir = os.path.join(path, 'L')
        self.right_dir = os.path.join(path, 'R')
          
        os.makedirs(self.left_dir)
        os.makedirs(self.right_dir)

    def __subdivide_camera_image(self, frame : np.ndarray) -> list:

        height, width = frame.shape[:2]
        mid_point = width // 2
        return [
                   frame[:, :mid_point],
                   frame[:, mid_point:]
               ]

    def __display_images(self, frame : np.ndarray):

        (
            left_image,
            right_image
        ) = self.__subdivide_camera_image(frame)                

        cv2.imshow('Left Image', left_image)
        cv2.imshow('Right Image', right_image)

    def __save_images(self, frame : np.ndarray):

        (
            left_image,
            right_image
        ) = self.__subdivide_camera_image(frame)                

        left_path = os.path.join(self.left_dir, f'{self.frame_count}.png')
        right_path = os.path.join(self.right_dir, f'{self.frame_count}.png')
        cv2.imwrite(left_path, left_image)
        cv2.imwrite(right_path, right_image)

if __name__ == '__main__':

    stream = Stream()
    stream.run()


