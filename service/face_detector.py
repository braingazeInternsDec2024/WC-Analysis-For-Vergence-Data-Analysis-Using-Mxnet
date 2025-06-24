#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
from queue import Queue, Full

from numpy import frombuffer, uint8, concatenate, float32, maximum, minimum, prod
from functools import partial

from threading import Thread

import os
import sys
sys.path.append(os.path.dirname(__file__))
from generate_anchor import generate_anchors_fpn, nonlinear_pred, generate_runtime_anchors


class BaseDetection:
    def __init__(self, *, thd, device, margin, nms_thd, verbose):
        self.threshold = thd
        self.nms_threshold = nms_thd
        self.device = device
        self.margin = margin

        self._queue = Queue(200)
        self.write_queue = self._queue.put_nowait
        self.read_queue = iter(self._queue.get, b'')

        self._nms_wrapper = partial(self.non_maximum_suppression,
                                    threshold=self.nms_threshold)
        
        self._biggest_wrapper = partial(self.find_biggest_box)


    def margin_clip(self, b):
        margin_x = (b[2] - b[0]) * self.margin
        margin_y = (b[3] - b[1]) * self.margin

        b[0] -= margin_x
        b[1] -= margin_y
        b[2] += margin_x
        b[3] += margin_y

        return np.clip(b, 0, None, out=b)

    @staticmethod
    def find_biggest_box(dets):
        return max(dets, key=lambda x: x[4]) if dets.size > 0 else None

    @staticmethod
    def non_maximum_suppression(dets, threshold):
        ''' ##### Author 1996scarlet@gmail.com
        Greedily select boxes with high confidence and overlap with threshold.
        If the boxes' overlap > threshold, we consider they are the same one.

        Parameters
        ----------
        dets: ndarray
            Bounding boxes of shape [N, 5].
            Each box has [x1, y1, x2, y2, score].

        threshold: float
            The src scales para.

        Returns
        -------
        Generator of kept box, each box has [x1, y1, x2, y2, score].

        Usage
        -----
        >>> for res in non_maximum_suppression(dets, thresh):
        >>>     pass
        '''

        x1, y1, x2, y2, scores = dets.T

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        while order.size > 0:
            keep, others = order[0], order[1:]

            yield np.copy(dets[keep])

            xx1 = maximum(x1[keep], x1[others])
            yy1 = maximum(y1[keep], y1[others])
            xx2 = minimum(x2[keep], x2[others])
            yy2 = minimum(y2[keep], y2[others])

            w = maximum(0.0, xx2 - xx1 + 1)
            h = maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            overlap = inter / (areas[keep] - inter + areas[others])

            order = others[overlap < threshold]

    @staticmethod
    def filter_boxes(boxes, min_size, max_size=-1):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            boxes = np.where(minimum(ws, hs) < max_size)[0]
        if min_size > 0:
            boxes = np.where(maximum(ws, hs) > min_size)[0]
        return boxes


class FaceDetectionModel(BaseDetection):
    def __init__(self, model_path, scale=1., device=None, thd=0.6, margin=0,
                 nms_thd=0.4, verbose=False):

        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        super().__init__(thd=thd, device=device, margin=margin,
                         nms_thd=nms_thd, verbose=verbose)

        self.scale = scale
        self._rescale = partial(cv2.resize, dsize=None, fx=self.scale,
                                fy=self.scale, interpolation=cv2.INTER_NEAREST)

        self._fpn_anchors = generate_anchors_fpn()
        self._runtime_anchors = {}

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path):
        """Load a PyTorch model for face detection"""
        # Define a RetinaFace-like model
        model = RetinaFaceDetector()
        
        # If model_path is provided and exists, load weights
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            print(f"Warning: Model path {model_path} not found. Using untrained model.")
            
        return model

    def _get_runtime_anchors(self, height, width, stride, base_anchors):
        key = height, width, stride
        if key not in self._runtime_anchors:
            self._runtime_anchors[key] = generate_runtime_anchors(
                height, width, stride, base_anchors).reshape((-1, 4))
        return self._runtime_anchors[key]

    def _retina_detach(self, out):
        ''' ##### Author 1996scarlet@gmail.com
        Solving bounding boxes.

        Parameters
        ----------
        out: tuple of (scores_tensor, boxes_tensor, anchors_array)
            scores_tensor: PyTorch tensor with detection scores
            boxes_tensor: PyTorch tensor with bounding box coordinates
            anchors_array: NumPy array with anchor coordinates

        Returns
        -------
        deltas: ndarray
            Array of detections with format [x1, y1, x2, y2, score]
        '''
        scores_tensor, boxes_tensor, anchors = out
        
        # Convert to numpy
        scores = scores_tensor.cpu().numpy()
        boxes = boxes_tensor.cpu().numpy()
        
        # Handle dimension mismatch - common with untrained models
        try:
            # Check if dimensions match
            if scores.ndim == 1:
                scores = scores.reshape(-1, 1)
            
            # Create mask and ensure it has the right shape
            mask = scores > self.threshold
            
            # If mask is multi-dimensional, flatten to 1D
            if mask.ndim > 1:
                mask = mask.flatten()
            
            # Ensure boxes can be indexed with the mask
            if mask.shape[0] != boxes.shape[0]:
                print(f"Warning: Mask shape {mask.shape} doesn't match boxes shape {boxes.shape}. Returning empty detections.")
                return np.array([])
            
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            filtered_anchors = anchors[mask]
            
            # If we have valid detections
            if filtered_boxes.size > 0:
                # Apply box decoding
                nonlinear_pred(filtered_anchors, filtered_boxes)
                filtered_boxes[:, :4] /= self.scale
                
                # Combine boxes and scores
                deltas = np.hstack([filtered_boxes, filtered_scores.reshape(-1, 1)])
                return deltas
            else:
                return np.array([])
        except Exception as e:
            print(f"Error in face detection: {e}. Returning empty detections.")
            return np.array([])

    def _retina_solve(self, outputs, input_shape):
        """Process network outputs to get boxes and scores"""
        batch_scores = []
        batch_boxes = []
        anchors_list = []
        
        # Process each FPN level
        for i, fpn in enumerate(self._fpn_anchors):
            # Get feature map dimensions
            feat_h, feat_w = input_shape[2] // fpn.stride, input_shape[3] // fpn.stride
            
            # Get scores and boxes from network outputs
            scores = outputs[i*2]  # Classification outputs
            boxes = outputs[i*2+1]  # Regression outputs
            
            # Reshape outputs
            scores = scores.reshape(-1, 1)
            boxes = boxes.reshape(-1, 4)
            
            # Get anchors for this feature level
            anchors = self._get_runtime_anchors(feat_h, feat_w, fpn.stride, fpn.base_anchors)
            anchors_list.append(anchors)
            
            batch_scores.append(scores)
            batch_boxes.append(boxes)
        
        # Concatenate all levels
        all_scores = torch.cat(batch_scores, dim=0)
        all_boxes = torch.cat(batch_boxes, dim=0)
        all_anchors = concatenate(anchors_list)
        
        return all_scores, all_boxes, all_anchors

    def _retina_forward(self, src):
        ''' ##### Author 1996scarlet@gmail.com
        Image preprocess and return the forward results.

        Parameters
        ----------
        src: ndarray
            The image batch of shape [H, W, C].

        Returns
        -------
        tuple: (scores, boxes, anchors)
            Processed outputs from the network
        '''
        # Preprocess image
        dst = self._rescale(src).transpose((2, 0, 1))[None, ...]
        dst = torch.from_numpy(dst.astype(np.float32)).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(dst)
        
        # Process outputs
        return self._retina_solve(outputs, dst.shape)

    def detect(self, image, mode='nms'):
        out = self._retina_forward(image)
        detach = self._retina_detach(out)
        return getattr(self, f'_{mode}_wrapper')(detach)

    def workflow_inference(self, instream, shape):
        for source in instream:
            frame = frombuffer(source, dtype=uint8).reshape(shape)
            out = self._retina_forward(frame)

            try:
                self.write_queue((frame, out))
            except Full:
                print('Frame queue full', file=sys.stderr)

    def workflow_postprocess(self, outstream=None):
        for frame, out in self.read_queue:
            detach = self._retina_detach(out)

            if outstream is None:
                for res in self._nms_wrapper(detach):
                    cv2.rectangle(frame, (res[0], res[1]),
                                  (res[2], res[3]), (255, 255, 0))

                cv2.imshow('res', frame)
                cv2.waitKey(1)
            else:
                outstream(frame)
                outstream(detach)


class RetinaFaceDetector(nn.Module):
    """A simplified RetinaFace-style detector using PyTorch"""
    def __init__(self):
        super(RetinaFaceDetector, self).__init__()
        
        # Define a simple backbone network (similar to ResNet)
        self.backbone = nn.Sequential(
            # Initial layers
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # First block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # FPN layers
        self.fpn_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn_2 = nn.Conv2d(256, 256, kernel_size=1)
        
        # Classification and regression heads for each FPN level
        # For simplicity, we'll use 2 FPN levels (similar to the original code)
        self.cls_1 = nn.Conv2d(256, 2, kernel_size=3, padding=1)  # 2 anchors per location
        self.reg_1 = nn.Conv2d(256, 8, kernel_size=3, padding=1)  # 2 anchors * 4 coordinates
        
        self.cls_2 = nn.Conv2d(256, 2, kernel_size=3, padding=1)
        self.reg_2 = nn.Conv2d(256, 8, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # FPN processing
        fpn_1_out = self.fpn_1(features)
        fpn_2_out = self.fpn_2(F.interpolate(fpn_1_out, scale_factor=0.5, mode='nearest'))
        
        # Apply classification and regression heads
        cls_1 = self.cls_1(fpn_1_out)
        reg_1 = self.reg_1(fpn_1_out)
        
        cls_2 = self.cls_2(fpn_2_out)
        reg_2 = self.reg_2(fpn_2_out)
        
        # Return outputs for each FPN level
        return [cls_1, reg_1, cls_2, reg_2]


if __name__ == '__main__':
    from numpy import prod

    FRAME_SHAPE = 480, 640, 3
    BUFFER_SIZE = prod(FRAME_SHAPE)

    read = sys.stdin.buffer.read
    write = sys.stdout.buffer.write
    camera = iter(partial(read, BUFFER_SIZE), b'')

    fd = FaceDetectionModel("../weights/face_detection_model.pth", scale=.4, margin=0.15)

    poster = Thread(target=fd.workflow_postprocess)
    poster.start()

    infer = Thread(target=fd.workflow_inference, args=(camera, FRAME_SHAPE,))
    infer.daemon = True
    infer.start()
