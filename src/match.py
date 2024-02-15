import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class ImageMatcher:
    def __init__(self, mode) -> None:
        self.mode = mode
    
    def match_template(self, template_path, image_path, method=cv2.TM_CCOEFF_NORMED):
        """
        Template matching is a technique for finding areas
        of an image that match (are similar) to a template image (patch).
        
        The function slides through image, compares the overlapped patches of size w*h with template image
        using the specified method and stores the comparison results in result that is the same size as image.
        
        Parameters
        ----------
        :param template_path: path to the template image
        :param image_path: path to the image
        :param method: method to use for template matching
        
        """
        
        image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(template_path, cv2.COLOR_BGR2GRAY)
        template_width, template_height = template.shape[0], template.shape[1]
        
        if template_width > image.shape[0] or template_height > image.shape[1]:
            raise ValueError("Template dimensions are larger than image dimensions") 
        
        result = cv2.matchTemplate(image, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        match_loc = (template_width, template_height) + min_loc + max_loc
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            
        bottom_right = (top_left[0] + template_width , top_left[1] + template_height)
        
        return match_loc, bottom_right, top_left, result, image
    
    def find_match_location(self, result):
        
        """
        Find the location of the match in the image
        
        Parameters
        
        :param result: result of the template matching

        """
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        return min_loc, max_loc
    
    def visualize_matches(self, result, image, match_location, method=cv2.TM_CCOEFF_NORMED):
        width, height, min_loc, max_loc = match_location[0], match_location[1], (match_location[2], match_location[3]), (match_location[4], match_location[5])
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            
        bottom_right = (top_left[0] + width , top_left[1] + height)
        
        cv2.rectangle(image, top_left, bottom_right, 255, 4)
        plt.subplot(121),plt.imshow(result, cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(image, cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show() 
        