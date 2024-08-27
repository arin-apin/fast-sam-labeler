from ultralytics import FastSAM
# from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch 
import cv2


class SAM:
    def __init__(self, confidence=0.4, device='cpu'):
        self.model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt
        self.device = device
        self.confidence = confidence
    
    def inference(self, filename, x, y):

        results = self.model.predict(source=filename, points=[[x, y]])
        masks_ = results[0].masks.data
        masks = [mask.cpu().numpy() for mask in masks_]
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
        # cv2.imshow('mask',img)
        return box
        


