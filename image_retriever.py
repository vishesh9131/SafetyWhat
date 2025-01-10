import os
import cv2
from datetime import datetime

class ImageRetriever:
    def __init__(self, output_dir='subobject_images'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def save_subobject_image(self, image, main_obj_name, main_obj_id, sub_obj_name, sub_obj_id):
        """Save a cropped image of a sub-object"""
        if image is None or image.size == 0:
            return
            
        # Create subdirectory for main object type if it doesn't exist
        main_obj_dir = os.path.join(self.output_dir, main_obj_name)
        if not os.path.exists(main_obj_dir):
            os.makedirs(main_obj_dir)
            
        # Generate filename with timestamp and IDs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{main_obj_name}_{main_obj_id}_{sub_obj_name}_{sub_obj_id}_{timestamp}.jpg"
        filepath = os.path.join(main_obj_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, image)