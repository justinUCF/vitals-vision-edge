# Agent_A/PerceptionProcessing/PerceptionEngine.py

"""
PerceptionEngine orchestrates YOLO detection + VLM captioning.
Handles enrichment with drone metadata (drone_id, location, captions).
"""

import numpy as np
from typing import List, Tuple, Optional
from PerceptionProcessing.YoloDetector import YoloDetector
from PerceptionProcessing.VLMCaptioner import VLMCaptioner
from PerceptionProcessing.Detection import Detection


class PerceptionEngine:
    def __init__(
        self,
        yolo_model_path: str,
        yolo_confidence: float = 0.5,
        yolo_iou: float = 0.45,
        caption_threshold: float = 0.7,
        device: str = "cpu",
        yolo_device: str = None,
        ollama_host: str = "http://localhost:11434"
    ):
        """
        Initialize PerceptionEngine with YOLO + VLM

        Args:
            yolo_model_path: Path to YOLO .pt model
            yolo_confidence: YOLO detection confidence threshold
            yolo_iou: YOLO IoU threshold for NMS
            caption_threshold: Only caption detections above this confidence
            device: Device for VLM captioner ('cuda' or 'cpu')
            yolo_device: Device for YOLO inference (defaults to device if None)
        """
        print("Initializing Perception Engine...")

        # Initialize YOLO detector
        self.yolo_detector = YoloDetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence,
            iou_threshold=yolo_iou,
            device=yolo_device if yolo_device is not None else device
        )

        # Initialize VLM captioner
        self.captioner = VLMCaptioner(device=device, ollama_host=ollama_host)
        
        self.caption_threshold = caption_threshold
        self.device = device
        
        print("Perception Engine ready")
    
    def process_image(
        self,
        image: np.ndarray,
        drone_id: Optional[int] = None,
        location: Optional[str] = None,
        generate_captions: bool = False,
        return_annotated: bool = True
    ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Process image from numpy array (cv2 frame)
        
        Args:
            image: Input image as numpy array (BGR format)
            drone_id: Optional drone ID for enrichment
            location: Optional location string
            generate_captions: Whether to generate VLM captions
            return_annotated: Whether to return annotated image
        
        Returns:
            Tuple of (detections_list, annotated_image)
        """
        # Run YOLO detection
        detections, annotated_img = self.yolo_detector.detect(
            image,
            return_annotated=return_annotated
        )

        if detections:
            print(f"Found {len(detections)} detection(s)")
        
        # Enrich with drone metadata
        for det in detections:
            det.enrich(drone_id=drone_id, location=location)
        
        # Generate captions if requested
        if generate_captions:
            detections = self._add_captions(image, detections)
        
        return detections, annotated_img
    
    def _add_captions(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> List[Detection]:
        """
        Add VLM captions to detections above confidence threshold

        Args:
            frame: Image as numpy array (BGR)
            detections: List of Detection objects

        Returns:
            Updated detections with captions
        """
        for i, det in enumerate(detections, 1):
            # Only caption high-confidence detections
            if det.confidence >= self.caption_threshold:
                print(f"Generating caption for detection {i}/{len(detections)}: {det.class_name}")
                
                try:
                    caption = self.captioner.caption_detection(
                        image=frame,
                        detection_bbox=det.bbox_pixels,
                        class_name=det.class_name,
                        confidence=det.confidence
                    )
                    
                    # Truncate for display
                    display_caption = caption[:80] + "..." if len(caption) > 80 else caption
                    print(f"  Caption: {det.class_name} (conf: {det.confidence:.2f}): {display_caption}")
                    
                    det.enrich(caption=caption)
                    
                except Exception as e:
                    print(f"Caption generation failed: {e}")
                    det.enrich(caption=None)
            else:
                print(f"Skipping caption for {det.class_name} (conf={det.confidence:.2f} < {self.caption_threshold})")
        
        return detections