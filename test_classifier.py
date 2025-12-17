import torch
from torch.utils.data import DataLoader
import yaml
import torch.nn.functional as F

from data.dataset import MIMICEyeDataset, collate_fn
from models.encoders import ImageEncoder, GazeEncoder, ImageOnlyClassifier
from models.attention import GazeGuidedFusion


def test_image_only_classifier():
    print("="*80)
    print("Testing Image-Only Classifier (Student Path)")
    print("="*80)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("\n1. Loading dataset...")
    dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='train',
        config=config
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, 
                           collate_fn=collate_fn, num_workers=0)
    batch = next(iter(dataloader))
    
    # Initialize models
    print("\n2. Initializing models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_encoder = ImageEncoder(model_name='vit_base_patch16_224', pretrained=True).to(device)
    gaze_encoder = GazeEncoder(spatial_hidden_dim=256, temporal_hidden_dim=256).to(device)
    gaze_fusion = GazeGuidedFusion(image_dim=768, gaze_dim=512).to(device)
    
    # NEW: Image-Only Classifier
    classifier = ImageOnlyClassifier(
        input_dim=768,      # Takes fused features
        hidden_dim=512,
        num_classes=2,      # Binary: normal vs abnormal
        dropout=0.3
    ).to(device)
    
    # Move data to device
    images = batch['images'].to(device)
    gaze_heatmaps = batch['gaze_heatmaps'].to(device)
    gaze_sequences = batch['gaze_sequences'].to(device)
    gaze_seq_lengths = batch['gaze_seq_lengths']
    labels = batch['labels'].to(device)
    
    print("\n3. Running full pipeline...")
    
    with torch.no_grad():
        # Encode
        patch_features, cls_token = image_encoder(images)
        gaze_weights, gaze_features = gaze_encoder(gaze_heatmaps, gaze_sequences, gaze_seq_lengths)
        
        # Fuse
        fused_features, attention_map = gaze_fusion(patch_features, cls_token, gaze_weights, gaze_features)
        print(f"  ✓ Fused features: {fused_features.shape}")
        
        # Classify
        logits = classifier(fused_features)
        print(f"  ✓ Logits: {logits.shape}")
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
        
        print(f"\n4. Classification results:")
        print(f"  Logits:\n{logits}")
        print(f"\n  Probabilities (class 0 / class 1):\n{probs}")
        print(f"\n  Predictions: {predictions}")
        print(f"  Ground truth: {labels}")
        
        # Calculate accuracy (will be ~50% since untrained)
        accuracy = (predictions == labels).float().mean().item()
        print(f"\n  Accuracy (untrained): {accuracy*100:.1f}%")
    
    print("\n" + "="*80)
    print("✓ Image-Only Classifier Test Complete!")
    print("="*80)
    print("\nPipeline verified:")
    print("  Image → Encoder → Gaze Fusion → Classifier → Predictions ✓")
    print("\nReady for training!")


if __name__ == '__main__':
    test_image_only_classifier()