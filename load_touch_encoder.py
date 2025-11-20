"""
Standalone example for loading and using the UniTouch touch encoder.
This loads ONLY the touch encoder without the full Touch-LLM model.
"""

import torch
import os
import ImageBind.data as data
from ImageBind.models.x2touch_model_part import x2touch, ModalityType


def load_touch_encoder(checkpoint_path='./last_new.ckpt', device='cuda'):
    """
    Load the pretrained touch encoder model.
    
    Args:
        checkpoint_path: Path to the touch encoder checkpoint (last_new.ckpt)
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        model: The loaded touch encoder model
    """
    # Verify checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Please download 'last_new.ckpt' from "
            f"https://huggingface.co/chfeng/Touch-LLM"
        )
    
    # Load the touch encoder model with pretrained weights
    # Note: The checkpoint path is hardcoded in x2touch() to './last_new.ckpt'
    model = x2touch(pretrained=True)
    model.eval()
    model = model.to(device)
    
    print(f"Touch encoder loaded successfully from {checkpoint_path}")
    return model


def extract_touch_embeddings(model, image_paths, device='cuda'):
    """
    Extract touch embeddings from touch images.
    
    Args:
        model: The touch encoder model
        image_paths: List of paths to touch images
        device: Device to run inference on
    
    Returns:
        embeddings: Touch embeddings (normalized, 1024-dim)
    """
    # Load and transform touch images
    touch_images = data.load_and_transform_vision_data(image_paths, device=device)
    
    # Extract embeddings using the proper modality type
    with torch.no_grad():
        outputs = model({ModalityType.TOUCH: touch_images})
        touch_embeddings = outputs[ModalityType.TOUCH]  # Shape: [batch_size, 1024]
    
    return touch_embeddings


def main():
    """Example usage of the touch encoder."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load the touch encoder
    model = load_touch_encoder(checkpoint_path='./last_new.ckpt', device=device)
    
    # 2. Example: Extract embeddings from a single touch image
    touch_image_path = "./touch_1/0000016865.jpg"  # Replace with your image path
    embeddings = extract_touch_embeddings(model, [touch_image_path], device=device)
    
    print(f"Touch embedding shape: {embeddings.shape}")
    print(f"Touch embedding (first 10 values): {embeddings[0, :10]}")
    
    # 3. Example: Batch processing multiple touch images
    # touch_image_paths = ["path/to/touch1.jpg", "path/to/touch2.jpg", "path/to/touch3.jpg"]
    # batch_embeddings = extract_touch_embeddings(model, touch_image_paths, device=device)
    # print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # 4. Example: Compare similarity between two touch images
    # touch_path_1 = "path/to/touch1.jpg"
    # touch_path_2 = "path/to/touch2.jpg"
    # emb1 = extract_touch_embeddings(model, [touch_path_1], device=device)
    # emb2 = extract_touch_embeddings(model, [touch_path_2], device=device)
    # similarity = (emb1 * emb2).sum()  # Cosine similarity (embeddings are normalized)
    # print(f"Touch similarity: {similarity.item()}")


if __name__ == "__main__":
    main()

