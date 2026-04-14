from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2
from typing import Union
# render_zoomin_rough_video(full_pose_seq, gaussians, "a bee is on the sunflower")

def zoom_image_by_focal_change(image: Union[Image.Image, np.ndarray, torch.Tensor], 
                              focal_1: float, 
                              focal_2: float) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Simulate the effect of focal length changes on image

    Focal length increase -> smaller FOV -> crop a smaller region from center and enlarge
    Focal length decrease -> larger FOV -> needs padding (here simply handled by shrinking image and padding)

    Args:
        image: input image, can be PIL.Image, numpy array, or torch.Tensor
        focal_1: original focal length
        focal_2: target focal length

    Returns:
        Processed image, same format as input
    """
    if focal_1 <= 0 or focal_2 <= 0:
        raise ValueError("Focal length must be greater than 0")
    
    # Record original input type
    input_type = type(image)
    
    # Convert to numpy array for processing
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
            img_array = image.permute(1, 2, 0).detach().cpu().numpy()
        else:  # HWC format
            img_array = image.detach().cpu().numpy()
        
        # If float type and in [0,1] range, convert to [0,255]
        if img_array.dtype == np.float32 or img_array.dtype == np.float64:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = image.copy()
    
    # Ensure uint8 type
    # if img_array.dtype != np.uint8:
    #     img_array = img_array.astype(np.uint8)
    
    h, w = img_array.shape[:2]
    
    # Compute zoom ratio
    zoom_ratio = focal_2 / focal_1
    
    if zoom_ratio > 1.0:
        # Zoom in: crop a smaller region from center then enlarge
        # Crop region dimensions
        crop_h = int(h / zoom_ratio)
        crop_w = int(w / zoom_ratio)
        
        # Ensure crop dimensions are even (for easier center computation)
        crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
        crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
        
        # Compute start position for center crop
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        
        # Crop center region
        if len(img_array.shape) == 3:  # Color image
            cropped = img_array[start_y:start_y+crop_h, start_x:start_x+crop_w, :]
        else:  # Grayscale image
            cropped = img_array[start_y:start_y+crop_h, start_x:start_x+crop_w]
        
        # Enlarge to original size
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
    elif zoom_ratio < 1.0:
        # Zoom out: shrink image and place at center, pad with black
        # Dimensions after shrinking
        new_h = int(h * zoom_ratio)
        new_w = int(w * zoom_ratio)
        
        # Shrink image
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create black background with same size as original image
        if len(img_array.shape) == 3:
            result = np.zeros((h, w, img_array.shape[2]), dtype=np.uint8)
        else:
            result = np.zeros((h, w), dtype=np.uint8)
        
        # Compute placement position (center)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        # Place the shrunk image at center
        if len(img_array.shape) == 3:
            result[start_y:start_y+new_h, start_x:start_x+new_w, :] = resized
        else:
            result[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    else:
        # zoom_ratio == 1.0, no processing needed
        result = img_array
    
    # Convert back to original format
    if input_type == Image.Image:
        return Image.fromarray(result)
    elif input_type == torch.Tensor:
        tensor_result = torch.from_numpy(result).float() / 255.0
        if image.dim() == 3 and image.shape[0] in [1, 3]:  # Originally CHW format
            tensor_result = tensor_result.permute(2, 0, 1)
        return tensor_result.to(image.device)
    else:
        return result


def zoom_image_crop_only(image: Union[Image.Image, np.ndarray, torch.Tensor], 
                        zoom_factor: float) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Simplified version: only supports zoom in (zoom_factor > 1), implemented via center crop

    Args:
        image: input image
        zoom_factor: zoom multiplier (> 1)

    Returns:
        Processed image, same format as input
    """
    if zoom_factor <= 1.0:
        print(f"Warning: zoom_factor should be greater than 1, current value: {zoom_factor}")
        return image
    
    return zoom_image_by_focal_change(image, focal_1=1.0, focal_2=zoom_factor)


# Test function
if __name__ == "__main__":
    # Test cases
    test_image = plt.imread("frames/saved_frames/input.png")
    plt.imsave("test_image.png", test_image)
    
    # Simulate focal length increasing from 100 to 200 (2x zoom)
    zoomed = zoom_image_by_focal_change(test_image, focal_1=100, focal_2=200)
    print(zoomed.shape, zoomed.max(), zoomed.min())
    plt.imsave("test_zoom_2x.png", (zoomed*255).astype(np.uint8))
    
    
    print("Test completed, generated test_zoom_2x.png and test_zoom_0.5x.png")