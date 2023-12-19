from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import torch 
import cv2

class CustomPFastSAMPrompt(FastSAMPrompt):
    def fast_show_mask(self, annotation):
        n, h, w = annotation.shape  # batch, height, width
        areas = np.sum(annotation, axis=(1, 2))
        annotation = annotation[np.argsort(areas)]
        index = (annotation != 0).argmax(axis=0)
        mask_image = np.expand_dims(annotation, -1) 
        show = np.zeros((h, w, 4))
        h_indices, w_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        show[h_indices, w_indices, :] = mask_image[indices]
        return show

    # original plot method returns matplotlib plots. i didn't find a more elegant way to get masks :)
    def plot(self, annotations):
        for ann in annotations:

            if ann.masks is not None:
                masks = ann.masks.data
                # if better_quality:
                if isinstance(masks[0], torch.Tensor):
                    masks = np.array(masks.cpu())
                for i, mask in enumerate(masks):
                    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    masks[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))

                res= self.fast_show_mask(masks)

                return res

class SAM:
    def __init__(self, confidence=0.4, device='cpu'):
        self.model = FastSAM('FastSAM-s.pt')  # or FastSAM-x.pt
        self.device = device
        self.confidence = confidence
    
    def inference(self, filename, x, y):
        everything_results = self.model(filename, device=self.device, retina_masks=True, imgsz=1024, conf=self.confidence, iou=0.9)
        prompt_process = CustomPFastSAMPrompt(filename, everything_results, device='cpu')
        ann = prompt_process.point_prompt(points=[[x, y]], pointlabel=[1])

        mask=prompt_process.plot(annotations=ann)
        bin_mask = (mask[:,:,0] != 0).astype('uint8')

        conns, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area=0
        box=None
        for i, cont in enumerate(conns):
            x, y, w, h = cv2.boundingRect(cont)
            if w*h>=max_area:
                max_area=w*h
                box=[x, y, w, h]
        return box
        


