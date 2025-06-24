import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class BaseSegmentation:
    def __init__(self, thd=0.05, device='cuda', verbose=False):
        self.thd = thd
        self.device = device
        self.verbose = verbose
        self.shape = np.array((96, 48))  # H, W
        self.eye_center = self.shape / 2  # x, y

    def calculate_gaze_mask_center(self, masks, method='pdc'):
        """ ##### Author 1996scarlet@gmail.com
        Obtain (x,y) coordinates given a set of N gaze heatmaps. 

        Parameters
        ----------
        masks : ndarray
            The predicted heatmaps, of shape [N, H, W, C]

        Returns
        -------
        points : ndarray
            Points of shape [N, 2] -> N * (x, y)

        Usage
        ----------
        >>> points = gs.calculate_gaze_mask_center(masks)
        [[49 23]
        [33 21]]
        >>> eye = cv2.circle(eye, tuple(points[n]), r, color)
        """
        N, H, W, _ = masks.shape

        def probability_density_center(masks, b=1e-7):
            masks[masks < self.thd] = 0
            masks_sum = np.sum(masks, axis=(1, 2))
            masks_sum += b

            x_sum = np.arange(W) @ np.sum(masks, axis=1)
            y_sum = np.arange(H) @ np.sum(masks, axis=2)

            points = np.hstack((x_sum, y_sum))
            return points/masks_sum

        if method == 'pdc':
            return probability_density_center(masks).astype(np.int32)
        else:
            indexes = np.argmax(masks.reshape((N, -1)), axis=1)
            return np.stack((indexes % W, indexes // W), axis=1)

    def plot_mask(self, src, masks, alpha=0.8, mono=True):
        draw = src.copy()

        for mask in masks:
            mask = np.repeat((mask > self.thd)[:, :, :], repeats=3, axis=2)
            if mono:
                draw = np.where(mask, 255, draw)
            else:
                color = np.random.random(3) * 255
                draw = np.where(mask, draw * (1 - alpha) + color * alpha, draw)

        return draw.astype('uint8')

    def draw_arrow(self, src, pupil_center, lengthen=5, color=(0, 125, 255), stroke=2, copy=False):
        if copy:
            draw = src.copy()
        else:
            draw = src

        H, W, C = draw.shape
        pt3 = self.eye_center + lengthen * (pupil_center - self.eye_center)

        scale = np.array([W, H]) / self.shape
        pt1 = (pt3 * scale).astype(np.int32)
        pt0 = (self.eye_center * scale).astype(np.int32)

        # cv2.drawMarker(draw, tuple(pt1), (255, 255, 0), markerType=cv2.MARKER_CROSS,
        #                              markerSize=2, thickness=2, line_type=cv2.LINE_AA)
        # cv2.drawMarker(draw, tuple(pt0), (255, 200, 200), markerType=cv2.MARKER_CROSS,
        #                              markerSize=2, thickness=2, line_type=cv2.LINE_AA)
        cv2.arrowedLine(draw, tuple(pt0), tuple(pt1), color, stroke)

        return draw


class GazeSegmentationModel(BaseSegmentation):
    def __init__(self, model_path, thd=0.05, device=None, verbose=False):
        super().__init__(thd, -1 if device is None else device, verbose)
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path):
        """Load a PyTorch model for gaze segmentation"""
        # Define a simple U-Net like architecture for gaze segmentation
        model = GazeSegmentationUNet(in_channels=3, out_channels=1)
        
        # If model_path is provided and exists, load weights
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            print(f"Warning: Model path {model_path} not found. Using untrained model.")
            
        return model

    def _get_gaze_mask_input(self, *eyes):
        """ ##### Author 1996scarlet@gmail.com
        Reduce mean and then stack the input eyes. 

        Parameters
        ----------
        eyes : ndarray
            The image batch of shape [N, H, W, C].

        Returns
        -------
        input_tensor : torch.Tensor
            PyTorch tensor of shape [N, C, H, W]. 
            [?, 3, 48, 96] for this model.

        Usage
        ----------
        >>> input_tensor = get_gaze_mask_input(eye1, eye2, ..., eyeN)
        """
        W, H = self.shape
        x = np.zeros((len(eyes), H, W, 3))

        np.stack([cv2.resize(e, (W, H)) for e in eyes], out=x)
        x -= 127.5
        x /= 127.5

        # Convert to PyTorch tensor and change from NHWC to NCHW format
        x = torch.from_numpy(x.transpose((0, 3, 1, 2))).float().to(self.device)
        return x

    def predict(self, *eyes):
        """ ##### Author 1996scarlet@gmail.com
        Predict blink classlabels and gaze masks of input eyes. 

        Parameters
        ----------
        eyes : ndarray
            The image batch of shape [N, H, W, C].

        Returns
        -------
        masks : ndarray
            Numpy ndarray of shape [N, H, W, 1]. 

        points : ndarray
            Numpy ndarray of shape [N, 2 -> (x, y)]. 

        Usage
        ----------
        >>> masks, points = gs.predict(left_eye, right_eye)
        """
        with torch.no_grad():
            x = self._get_gaze_mask_input(*eyes)
            output = self.model(x)
            
            # Convert output to numpy for post-processing
            masks = output.cpu().numpy().transpose((0, 2, 3, 1))
            
            points = self.calculate_gaze_mask_center(masks)
            points = points.astype(np.float32)
            
            for i, e in enumerate(eyes):
                points[i] *= e.shape[1]
                points[i] /= 96.0
                
            return masks, points


class GazeSegmentationUNet(nn.Module):
    """A simple U-Net architecture for gaze segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super(GazeSegmentationUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        # Decoder
        self.dec3 = self._conv_block(128, 64)
        self.dec2 = self._conv_block(64 + 64, 32)
        self.dec1 = self._conv_block(32 + 32, 16)
        
        # Final layer
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Decoder with skip connections
        d3 = self.dec3(e3)
        d3 = self.upsample(d3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d2 = self.upsample(d2)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        # Final output
        out = self.final(d1)
        return torch.sigmoid(out)  # Apply sigmoid for mask output
