"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for ConvNeXt-Tiny.
Targets the last convolutional block in features[7].
Produces an explanatory visual overlay highlighting regions influencing model classification.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
import io
import base64


class ConvNeXtGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module = None):
        self.model = model
        self.model.eval()
        
        # In torchvision convnext_tiny, features[7] is the 4th stage, features[7][-1] is the last CNBlock
        if target_layer is None:
            self.target_layer = self.model.features[7][-1]
        else:
            self.target_layer = target_layer
            
        self.activations = None
        self.gradients = None
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generates normalized (0 to 1) 2D numpy heatmap for the specified class index.
        """
        # Ensure gradients are enabled for this computation
        self.model.zero_grad()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
            
        # Target score for class
        score = logits[0, class_idx]
        score.backward()
        
        # Gradients: shape (1, C, H, W)
        grads = self.gradients
        acts = self.activations
        
        if grads is None or acts is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")
            
        # Global average pooling of gradients over spatial dimensions
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only features with positive influence
        
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam


def overlay_gradcam(original_img: Image.Image, heatmap_2d: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlays a 2D heatmap on the original PIL image using the 'jet' colormap.
    Returns composite PIL Image in RGB format.
    """
    orig_w, orig_h = original_img.size
    
    # Resize heatmap to match image dimensions
    cam_pil = Image.fromarray((heatmap_2d * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)
    cam_norm = np.array(cam_pil) / 255.0
    
    # Apply JET colormap (returns RGBA [0..1])
    try:
        import matplotlib
        colormap = matplotlib.colormaps["jet"]
    except Exception:
        colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(cam_norm)[:, :, :3]  # RGB, float 0..1
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    
    # Convert original to RGB numpy
    orig_np = np.array(original_img.convert("RGB"))
    
    # Blend images
    blended = (alpha * colored_heatmap + (1.0 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Converts a PIL Image to a data-URI base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=90)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() in ("JPG", "JPEG") else "image/png"
    return f"data:{mime};base64,{b64_str}"
