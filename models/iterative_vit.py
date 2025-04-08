import torch
import torch.nn as nn
from .vit import VisionTransformer, PatchEmbedding

class IterativeViT(VisionTransformer):
    def __init__(self, num_iterative_tokens=3, num_iterations=5, layer_idx=-2, **kwargs):
        """
        Args:
            num_iterative_tokens (int): Number of iterative tokens to add
            num_iterations (int): Number of times to process the model
            layer_idx (int): Which transformer layer's output to use for iterative tokens
                            -1 means last layer, -2 means second to last, etc.
                            Or use positive index (0-based) from first layer
            **kwargs: Arguments passed to parent VisionTransformer
        """
        super().__init__(**kwargs)
        self.emb_size = kwargs.get('emb_size', 768)
        
        # Store iteration parameters
        self.num_iterative_tokens = num_iterative_tokens
        self.num_iterations = num_iterations
        
        # Convert layer_idx to positive index
        if layer_idx < 0:
            layer_idx = len(self.blocks) + layer_idx
        self.layer_idx = layer_idx
        
        # Create iterative tokens (not trainable)
        self.iterative_tokens = nn.Parameter(
            torch.zeros(1, num_iterative_tokens, self.emb_size),
            requires_grad=False
        )
        
        # Create learnable step embeddings for each iteration
        self.step_embeddings = nn.Parameter(
            torch.randn(1, num_iterations, self.emb_size)
        )
        
        # Modify the patch embedding to accommodate iterative tokens
        self.patch_embed = IterativePatchEmbedding(
            in_channels=kwargs.get('in_channels', 3),
            patch_size=kwargs.get('patch_size', 16),
            emb_size=kwargs.get('emb_size', 768),
            img_size=kwargs.get('img_size', 224),
            num_iterative_tokens=num_iterative_tokens
        )

    def forward(self, x):
        # Initialize iterative tokens
        B = x.size(0)
        iterative_tokens = self.iterative_tokens.expand(B, -1, -1)
        
        # Store outputs for each iteration
        all_outputs = []
        
        for i in range(self.num_iterations):
            # Get the step embedding for the current iteration
            step_emb = self.step_embeddings[:, i:i+1].expand(B, -1, -1)
            
            # Get patch embeddings and concatenate tokens
            x_embed = self.patch_embed(x, iterative_tokens, step_emb)
            
            # Process through transformer blocks up to specified layer
            x_out = x_embed
            for j, block in enumerate(self.blocks):
                x_out = block(x_out)
                if j == self.layer_idx and i < self.num_iterations - 1:
                    # Extract iterative tokens from this layer's output
                    # Assumes step token is the last one
                    with torch.no_grad():  # Gradient cut
                        iterative_tokens = x_out[:, -(self.num_iterative_tokens + 1):-1]
            
            x_out_norm = self.norm(x_out)
            
            # Get CLS token and pass through head
            cls_token = x_out_norm[:, 0]
            output = self.head(cls_token)
            
            # Store the output
            all_outputs.append(output)
        
        # Return list of outputs from all iterations
        return all_outputs

class IterativePatchEmbedding(PatchEmbedding):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, num_iterative_tokens=3):
        super().__init__(in_channels, patch_size, emb_size, img_size)
        # Only create position embeddings for iterative tokens
        self.iterative_pos_embed = nn.Parameter(
            torch.randn(1, num_iterative_tokens, emb_size)
        )
    
    def forward(self, x, iterative_tokens, step_token):
        # Get patch embeddings (includes position embeddings for patches and CLS token)
        x = super().forward(x)
        
        # Add position embeddings to iterative tokens
        iterative_tokens = iterative_tokens + self.iterative_pos_embed
        
        # Concatenate everything: CLS, Patches, Iterative Tokens, Step Token
        x = torch.cat([x, iterative_tokens, step_token], dim=1)
        
        return x 