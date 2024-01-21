from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch 
import cv2

class CustomPFastSAMPrompt(FastSAMPrompt):

    def get_masks(self):
        if self.results is None:
            return []

        masks = self.results[0].masks.data
        return [mask.cpu().numpy() for mask in masks]

class SAM:
    def __init__(self, confidence=0.4, device='cpu'):
        self.model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt
        self.device = device
        self.confidence = confidence
    
    def inference(self, filename, x, y):
        everything_results = self.model(filename, device=self.device, retina_masks=True, imgsz=1024, conf=self.confidence, iou=0.9)
        prompt_process = CustomPFastSAMPrompt(filename, everything_results, device='cpu')
        ann = prompt_process.point_prompt(points=[[x, y]], pointlabel=[1])
        masks = prompt_process.get_masks()
        
        if masks:
            mask = masks[0]
        bin_mask = (mask != 0).astype('uint8')

        conns, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area=0
        box=None
        for i, cont in enumerate(conns):
            x, y, w, h = cv2.boundingRect(cont)
            if w*h>=max_area:
                max_area=w*h
                box=[x, y, w, h]
        return box
        


