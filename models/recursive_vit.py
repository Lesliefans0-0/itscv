import torch
import torch.nn as nn
from .vit import VisionTransformer, PatchEmbedding

class RecursiveViT(VisionTransformer):
    def __init__(self, num_iterations=5, layer_idx=-2, **kwargs):
        """
        Args:
            num_iterations (int): Number of times to process the model
            layer_idx (int): Which transformer layer's output to use for recursive processing
                            -1 means last layer, -2 means second to last, etc.
                            Or use positive index (0-based) from first layer
            **kwargs: Arguments passed to parent VisionTransformer
        """
        super().__init__(**kwargs)
        self.emb_size = kwargs.get('emb_size', 768)
        
        # Store iteration parameters
        self.num_iterations = num_iterations
        
        # Convert layer_idx to positive index
        if layer_idx < 0:
            layer_idx = len(self.blocks) + layer_idx
        self.layer_idx = layer_idx

    def forward(self, x):
        # Initialize with just CLS token
        B = x.size(0)
        x_embed = self.patch_embed(x)
        
        # Store outputs for each iteration
        all_outputs = []
        
        for i in range(self.num_iterations):
            # Process through transformer blocks up to specified layer
            x_out = x_embed
            for j, block in enumerate(self.blocks):
                x_out = block(x_out)
                if j == self.layer_idx and i < self.num_iterations - 1:
                    # Use all tokens from this layer's output for next iteration
                    # Keep CLS token as first token
                    with torch.no_grad():  # Gradient cut
                        x_embed = x_out
            
            x_out_norm = self.norm(x_out)
            
            # Get CLS token and pass through head
            cls_token = x_out_norm[:, 0]
            output = self.head(cls_token)
            
            # Store the output
            all_outputs.append(output)
        
        # Return list of outputs from all iterations
        return all_outputs
