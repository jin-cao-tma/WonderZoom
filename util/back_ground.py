
import torch
import numpy as np
from PIL import Image
import cv2

# GroundingDINO imports
import torch
import numpy as np
from PIL import Image
import cv2
import sys 
sys.path.append('./Grounded-Segment-Anything/')
# GroundingDINO imports
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# SAM imports
from segment_anything import build_sam, SamPredictor
import PIL 


class GroundedSAMSegmentationModel:
    """
    Segmentation model based on GroundingDINO + SAM
    Supports text-prompted object detection and segmentation
    """

    def __init__(self,
                 grounding_config_path='./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                 grounding_checkpoint_path='./Grounded-Segment-Anything/groundingdino_swint_ogc.pth',
                 sam_checkpoint_path='./Grounded-Segment-Anything/sam_vit_h_4b8939.pth',
                 device='cuda'):
        """
        Initialize the segmentation model

        Args:
            grounding_config_path: GroundingDINO config file path
            grounding_checkpoint_path: GroundingDINO model weights path
            sam_checkpoint_path: SAM model weights path
            device: Device type ('cuda' or 'cpu')
        """
        self.device = device
        
        # Model initialization flags
        self.grounding_model = None
        self.sam_predictor = None

        # Store model paths
        self.grounding_config_path = grounding_config_path
        self.grounding_checkpoint_path = grounding_checkpoint_path
        self.sam_checkpoint_path = sam_checkpoint_path
        
        # Image preprocessing transforms
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.load_models()


    def load_models(self):
        """Load GroundingDINO and SAM models"""
        if self.grounding_model is None:
            print("Loading GroundingDINO model...")
            self.grounding_model = self._load_grounding_model()
            self.grounding_model.to(self.device)

        if self.sam_predictor is None:
            print("Loading SAM model...")
            self.sam_predictor = self._load_sam_model()

    def _load_grounding_model(self):
        """Load GroundingDINO model"""
        args = SLConfig.fromfile(self.grounding_config_path)
        args.device = self.device
        model = build_model(args)
        
        checkpoint = torch.load(self.grounding_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(f"GroundingDINO load result: {load_res}")

        model.eval()
        return model

    def _load_sam_model(self):
        """Load SAM model"""
        sam = build_sam(checkpoint=self.sam_checkpoint_path)
        sam.to(device=self.device)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    def _process_image(self, image):
        """Preprocess input image"""
        if isinstance(image, str):
            # If it is a file path, read the image
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If it is a numpy array, convert to PIL image
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL image, numpy array, or image file path")

        return image

    def get_combined_foreground_mask(self,image_path, foreground_words, box_threshold=0.3, text_threshold=0.25, kernel_size=9):
        """
    Given an image path and a list of foreground object words, use GroundedSAMSegmentationModel to extract and merge masks for all objects.
    Returns a single binary mask (torch.Tensor, shape [H, W], dtype=bool) where True indicates foreground.
    """
    # global grounded_sam
        grounded_sam = self 
        if grounded_sam is None:
            grounded_sam = GroundedSAMSegmentationModel()
        # Load the image
        image = Image.open(image_path).convert('RGB')
        masks = []
        for word in foreground_words:
            result = grounded_sam.segment(image, word, box_threshold=box_threshold, text_threshold=text_threshold)
            if result['masks'].shape[0] > 0:
                # Each mask is [1, H, W], take the first mask for this word
                for m in result['masks']:
                    masks.append(m.squeeze().bool())
        if not masks:
            # No masks found, return all False
            return torch.zeros((image.height, image.width), dtype=torch.bool)
        # Merge all masks (logical OR)
        combined_mask = masks[0].clone()
        masks = [mask for mask in masks if mask.float().mean() > 0.003]
        for mask in masks[1:]:
            combined_mask = combined_mask | mask
            
        mask_np = combined_mask.cpu().numpy().astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=1)
        dilated_mask_tensor = torch.from_numpy(dilated_mask).bool()
        return dilated_mask_tensor, combined_mask, masks
    
    def _get_grounding_output(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """Use GroundingDINO to get detection results"""
        # Preprocess text prompt
        caption = text_prompt.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        # Image preprocessing - ensure tensor is on CUDA
        transformed_image = self.transform(image, None)[0].to(self.device)

        # Model inference
        with torch.no_grad():
            outputs = self.grounding_model(transformed_image.unsqueeze(0), captions=[caption])

        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

        # Filter low-confidence results
        filt_mask = logits.max(dim=1)[0] > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]
        
        # Get predicted phrases
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)

        pred_phrases = []
        scores = []

        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            pred_phrases.append(pred_phrase)
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases

    def segment(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Segment an image

        Args:
            image: Input image (PIL.Image, numpy.ndarray, or file path)
            text_prompt: Text prompt describing the object to segment
            box_threshold: Bounding box threshold
            text_threshold: Text threshold

        Returns:
            dict: Dictionary containing the following keys:
                - 'masks': Segmentation masks (torch.Tensor)
                - 'boxes': Detection boxes (torch.Tensor)
                - 'phrases': Detected phrases (list)
                - 'scores': Confidence scores (torch.Tensor)
        """
        # Ensure models are loaded
        # self.load_models()

        # Process input image
        image_pil = self._process_image(image)
        image_np = np.array(image_pil)
        
        # Get image dimensions
        W, H = image_pil.size

        # Use GroundingDINO for detection
        boxes_filt, scores, pred_phrases = self._get_grounding_output(
            image_pil, text_prompt, box_threshold, text_threshold
        )
        
        if len(boxes_filt) == 0:
            print("No target object detected")
            return {
                'masks': torch.empty(0, 1, H, W),
                'boxes': torch.empty(0, 4),
                'phrases': [],
                'scores': torch.empty(0)
            }
        
        # Process detection box coordinates (convert from relative to absolute coordinates) - ensure tensor device consistency
        scale_tensor = torch.Tensor([W, H, W, H]).to(boxes_filt.device)
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * scale_tensor
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2  # Convert center point to top-left corner
            boxes_filt[i][2:] += boxes_filt[i][:2]      # Convert width/height to bottom-right corner

        # Use SAM for segmentation
        self.sam_predictor.set_image(image_np)
        
        # Convert bounding box format for SAM
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            boxes_filt, image_np.shape[:2]
        ).to(self.device)
        
        # Get segmentation masks
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        return {
            'masks': masks.cpu(),
            'boxes': boxes_filt,
            'phrases': pred_phrases,
            'scores': scores
        }
    
    def get_mask_image(self, masks, original_size):
        """
        Convert masks to a visualization image

        Args:
            masks: Segmentation masks
            original_size: Original image dimensions (width, height)

        Returns:
            PIL.Image: Mask image
        """
        if len(masks) == 0:
            return Image.new('L', original_size, 0)
        
        # Merge all masks
        combined_mask = torch.zeros(masks.shape[-2:], dtype=torch.bool)
        for mask in masks:
            combined_mask |= mask[0]

        # Convert to PIL image
        mask_array = combined_mask.cpu().numpy().astype(np.uint8) * 255
        mask_image = Image.fromarray(mask_array, mode='L')
        
        return mask_image
    
    def segment_and_visualize(self, image, text_prompt, box_threshold=0.3, text_threshold=0.25):
        """
        Segment an image and return visualization results

        Args:
            image: Input image
            text_prompt: Text prompt
            box_threshold: Bounding box threshold
            text_threshold: Text threshold

        Returns:
            dict: Dictionary containing segmentation results and visualization image
        """
        # Perform segmentation
        result = self.segment(image, text_prompt, box_threshold, text_threshold)
        
        # Process input image
        image_pil = self._process_image(image)

        # Generate mask image
        mask_image = self.get_mask_image(result['masks'], image_pil.size)
        
        # Add visualization results
        result['mask_image'] = mask_image
        result['original_image'] = image_pil

        return result




import torch
import numpy as np
from PIL import Image
from typing import Union, Optional
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0

# Global model instance
_inpaint_pipeline = None

def load_inpaint_model(model_path: str = "stabilityai/stable-diffusion-2-inpainting",
                      device: str = "cuda",
                      torch_dtype: torch.dtype = torch.bfloat16) -> bool:
    """
    Load the Stable Diffusion Inpainting model.
    
    Args:
        model_path: Path to the model checkpoint or HuggingFace model name
        device: Device to run the model on ("cuda" or "cpu")
        torch_dtype: Data type for the model
        
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    global _inpaint_pipeline
    
    try:
        print(f"🔄 Loading inpainting model from: {model_path}")
        
        _inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            safety_checker=None,
            torch_dtype=torch_dtype,
        ).to(device)
        
        # Configure scheduler
        _inpaint_pipeline.scheduler = DDIMScheduler.from_config(_inpaint_pipeline.scheduler.config)
        
        # Set attention processors for better performance
        _inpaint_pipeline.unet.set_attn_processor(AttnProcessor2_0())
        _inpaint_pipeline.vae.set_attn_processor(AttnProcessor2_0())
        
        print("✅ Inpainting model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load inpainting model: {e}")
        _inpaint_pipeline = None
        return False

def inpaint_image(image: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
                 mask: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
                 prompt: str = "",
                 negative_prompt: str = "",
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 output_type: str = "tensor") -> Union[torch.Tensor, PIL.Image.Image]:
    """
    Perform inpainting on an image.
    
    Args:
        image: Input image to inpaint. Can be:
            - torch.Tensor: [B, C, H, W] or [C, H, W] format, values in [0, 1]
            - PIL.Image.Image: RGB image
            - np.ndarray: [H, W, C] format, values in [0, 255]
        mask: Inpainting mask. Can be:
            - torch.Tensor: [B, 1, H, W] or [1, H, W] format, values in [0, 1] (1 = inpaint area)
            - PIL.Image.Image: Grayscale image (white = inpaint area)
            - np.ndarray: [H, W] format, values in [0, 255] (255 = inpaint area)
        prompt: Text prompt for inpainting
        negative_prompt: Negative text prompt
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        output_type: Output format ("tensor" or "pil")
        
    Returns:
        Union[torch.Tensor, PIL.Image.Image]: Inpainted image
        
    Raises:
        RuntimeError: If model is not loaded
    """
    global _inpaint_pipeline
    
    if _inpaint_pipeline is None:
        raise RuntimeError("Model not loaded. Call load_inpaint_model() first.")
    
    # Preprocess inputs
    image_tensor, mask_tensor = _preprocess_inputs(image, mask, height, width)
    
    # Convert to PIL for the pipeline
    image_pil = _tensor_to_pil(image_tensor)
    mask_pil = _tensor_to_pil(mask_tensor, is_mask=True)
    
    # Run inpainting
    with torch.no_grad():
        result = _inpaint_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height or image_tensor.shape[-2],
            width=width or image_tensor.shape[-1],
            output_type="pil" if output_type == "pil" else "pt"
        )
    
    # Process output
    if output_type == "pil":
        return result.images[0]
    else:
        # When output_type="pt", result.images[0] is already a tensor
        return result.images[0]

def _preprocess_inputs(image, mask, height=None, width=None):
    """Preprocess image and mask inputs"""
    # Process image
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            raise ValueError("Image array should be [H, W, 3] format")
    elif isinstance(image, PIL.Image.Image):
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("Image tensor should be [B, C, H, W] or [C, H, W] format")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Process mask
    if isinstance(mask, np.ndarray):
        if mask.ndim == 2:
            mask = torch.from_numpy(mask).float() / 255.0
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Mask array should be [H, W] format")
    elif isinstance(mask, PIL.Image.Image):
        mask = torch.from_numpy(np.array(mask.convert('L'))).float() / 255.0
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif isinstance(mask, torch.Tensor):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
        if mask.ndim != 4:
            raise ValueError("Mask tensor should be [B, 1, H, W], [1, H, W], or [H, W] format")
    else:
        raise ValueError(f"Unsupported mask type: {type(mask)}")
    
    # Resize if needed
    if height is not None or width is not None:
        target_height = height or image.shape[-2]
        target_width = width or image.shape[-1]
        
        # Resize image
        if image.shape[-2] != target_height or image.shape[-1] != target_width:
            image = torch.nn.functional.interpolate(
                image, size=(target_height, target_width), mode='bilinear', align_corners=False
            )
        
        # Resize mask
        if mask.shape[-2] != target_height or mask.shape[-1] != target_width:
            mask = torch.nn.functional.interpolate(
                mask, size=(target_height, target_width), mode='nearest'
            )
    
    # Ensure values are in correct range
    image = torch.clamp(image, 0, 1)
    mask = torch.clamp(mask, 0, 1)
    
    return image, mask

def _tensor_to_pil(tensor, is_mask=False):
    """Convert tensor to PIL Image"""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    if is_mask:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(tensor.cpu().numpy(), mode='L')
    else:
        tensor = tensor.permute(1, 2, 0)
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(tensor.cpu().numpy(), mode='RGB')

def _pil_to_tensor(pil_image):
    """Convert PIL Image to tensor"""
    array = np.array(object=pil_image.convert('RGB'))
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0) 
# Usage example
def inpaint_background(image_path, mask, background_prompt, num_inference_steps=50, output_type="pil"):
    """
    Inpaint the background of an image using a given mask and background prompt.
    Args:
        image_path (str): Path to the input image
        mask (torch.Tensor): Binary mask (H, W, bool), True = inpaint area
        background_prompt (str): Text prompt for inpainting
        num_inference_steps (int): Number of diffusion steps
        output_type (str): 'pil' or 'tensor'
    Returns:
        PIL.Image or torch.Tensor: The inpainted image
    """
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    # Call inpaint_image (assumes model is already loaded globally)
    result = inpaint_image(
        image=image,
        mask=mask,
        prompt=background_prompt,
        num_inference_steps=num_inference_steps,
        output_type=output_type
    )
    return result