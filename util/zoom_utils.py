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


class GroundedSAMSegmentationModel:
    """
    Segmentation model based on GroundingDINO + SAM
    Supports text-prompted object detection and segmentation
    """

    def __init__(self,
                 grounding_config_path='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                 grounding_checkpoint_path='groundingdino_swint_ogc.pth',
                 sam_checkpoint_path='sam_vit_h_4b8939.pz',
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