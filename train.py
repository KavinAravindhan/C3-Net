import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
import json
from datetime import datetime
import wandb
from dotenv import load_dotenv

from data.dataset import MIMICEyeDataset, collate_fn
from models.encoders import ImageEncoder, GazeEncoder, ImageOnlyClassifier, GazePredictor
from models.attention import GazeGuidedFusion

# Load environment variables
load_dotenv()

class C3NetTrainer:
    """
    Training system for C3-Net (Cognitive Causal Chain Network)
    Current implementation: Basic student path training
    - Image encoder + Gaze encoder + Fusion + Classifier
    - Single classification loss
    """
    
    def __init__(self, config):
        self.config = config

        if config['training']['device'] == 'cuda' and torch.cuda.is_available():
            gpu_id = config['training']['gpu_id']
            self.device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        print("="*80)
        print("Initializing C3-Net Training")
        print("="*80)
        
        # Initialize models
        print("\n1. Loading models...")
        self.image_encoder = ImageEncoder(
            model_name=config['model']['image_encoder']['type'],
            pretrained=config['model']['image_encoder']['pretrained']
        ).to(self.device)
        
        self.gaze_encoder = GazeEncoder(
            spatial_hidden_dim=config['model']['gaze_encoder']['spatial_hidden_dim'],
            temporal_hidden_dim=config['model']['gaze_encoder']['temporal_hidden_dim'],
            lstm_layers=config['model']['gaze_encoder']['lstm_layers'],
            dropout=config['model']['gaze_encoder']['dropout']
        ).to(self.device)
        
        self.gaze_fusion = GazeGuidedFusion(
            image_dim=config['model']['image_encoder']['embed_dim'],
            gaze_dim=config['model']['gaze_encoder']['spatial_hidden_dim'] + 
                     config['model']['gaze_encoder']['temporal_hidden_dim']
        ).to(self.device)
        
        self.classifier = ImageOnlyClassifier(
            input_dim=config['model']['image_encoder']['embed_dim'],
            hidden_dim=512,
            num_classes=2,
            dropout=0.3
        ).to(self.device)

        self.gaze_predictor = GazePredictor(
            input_dim=config['model']['image_encoder']['embed_dim'],
            hidden_dim=256,
            num_patches=196
        ).to(self.device)
        
        # Combine all parameters for optimizer
        self.all_params = (
            list(self.image_encoder.parameters()) +
            list(self.gaze_encoder.parameters()) +
            list(self.gaze_fusion.parameters()) +
            list(self.classifier.parameters()) +
            list(self.gaze_predictor.parameters())
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.all_params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # # Loss function
        # self.criterion = nn.CrossEntropyLoss()
        # self.gaze_criterion = nn.MSELoss()

        # Loss function with class weights
        # Normal (33%) gets higher weight, Abnormal (67%) gets lower weight
        # Inverse frequency weighting: 1/freq normalized
        class_weights = torch.tensor([1.0/0.33, 1.0/0.67]).to(self.device)
        # Normalize to sum to 2.0 (standard practice)
        class_weights = class_weights / class_weights.sum() * 2.0
        print(f"  Class weights: Normal={class_weights[0]:.3f}, Abnormal={class_weights[1]:.3f}")

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.gaze_criterion = nn.KLDivLoss(reduction='batchmean')

        # Loss weights (normalized to sum to 1.0)
        self.lambda_gaze_pred = config['training']['lambda_gaze_pred']
        self.lambda_classification = 1.0 - self.lambda_gaze_pred
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
        # Initialize wandb
        wandb.init(
            entity="ai4vs",
            project="c3_net",
            config={
                "learning_rate": config['training']['learning_rate'],
                "batch_size": config['training']['batch_size'],
                "num_epochs": config['training']['num_epochs'],
                "lambda_classification": self.lambda_classification,
                "lambda_gaze_pred": self.lambda_gaze_pred,
                "image_encoder": config['model']['image_encoder']['type'],
                "gaze_spatial_dim": config['model']['gaze_encoder']['spatial_hidden_dim'],
                "gaze_temporal_dim": config['model']['gaze_encoder']['temporal_hidden_dim'],
            }
        )
        
        print(f"\n✓ Models initialized on {self.device}")
        print(f"✓ Total trainable parameters: {sum(p.numel() for p in self.all_params):,}")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.image_encoder.train()
        self.gaze_encoder.train()
        self.gaze_fusion.train()
        self.classifier.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(self.device)
            gaze_heatmaps = batch['gaze_heatmaps'].to(self.device)
            gaze_sequences = batch['gaze_sequences'].to(self.device)
            gaze_seq_lengths = batch['gaze_seq_lengths']
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # 1. Encode image
            patch_features, cls_token = self.image_encoder(images)
            
            # 2. Encode gaze
            gaze_weights, gaze_features = self.gaze_encoder(
                gaze_heatmaps, gaze_sequences, gaze_seq_lengths
            )
            
            # 3. Fuse with gaze guidance
            fused_features, attention_map = self.gaze_fusion(
                patch_features, cls_token, gaze_weights, gaze_features
            )
            
            # # 4. Predict gaze from image (auxiliary task)
            # predicted_gaze = self.gaze_predictor(patch_features)  # [B, 196]
            # # Reshape to match gaze_weights [B, 196, 1]
            # target_gaze = gaze_weights.squeeze(-1)  # [B, 196]
            
            # 4. Predict gaze from image (auxiliary task)
            predicted_gaze = self.gaze_predictor(patch_features)  # [B, 196]
            # For KL divergence: apply log_softmax to predictions, target already normalized
            predicted_gaze_log = torch.log_softmax(predicted_gaze, dim=1)  # [B, 196]
            target_gaze = gaze_weights.squeeze(-1)  # [B, 196]

            # 5. Classify
            logits = self.classifier(fused_features)

            # # Compute losses
            # classification_loss = self.criterion(logits, labels)
            # gaze_prediction_loss = self.gaze_criterion(predicted_gaze, target_gaze)

            # Compute losses
            classification_loss = self.criterion(logits, labels)
            gaze_prediction_loss = self.gaze_criterion(predicted_gaze_log, target_gaze)

            # Total loss
            loss = (self.lambda_classification * classification_loss + 
                    self.lambda_gaze_pred * gaze_prediction_loss)
            
            # Track individual losses (optional, for monitoring)
            if batch_idx == 0:  # Print first batch of epoch
                print(f"\n  Losses: cls={classification_loss.item():.4f}, gaze={gaze_prediction_loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.all_params, max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

            # Log to wandb (per batch - optional, can be noisy)
            wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_cls_loss": classification_loss.item(),
                    "train/batch_gaze_loss": gaze_prediction_loss.item(),
                })
            
            # # Log to wandb every 10 batches
            # if batch_idx % 10 == 0:
            #     wandb.log({
            #         "train/batch_loss": loss.item(),
            #         "train/batch_cls_loss": classification_loss.item(),
            #         "train/batch_gaze_loss": gaze_prediction_loss.item(),
            #     })
                
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.image_encoder.eval()
        self.gaze_encoder.eval()
        self.gaze_fusion.eval()
        self.classifier.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            for batch in pbar:
                images = batch['images'].to(self.device)
                gaze_heatmaps = batch['gaze_heatmaps'].to(self.device)
                gaze_sequences = batch['gaze_sequences'].to(self.device)
                gaze_seq_lengths = batch['gaze_seq_lengths']
                labels = batch['labels'].to(self.device)
                
                # Forward pass (same as training)
                patch_features, cls_token = self.image_encoder(images)
                gaze_weights, gaze_features = self.gaze_encoder(
                    gaze_heatmaps, gaze_sequences, gaze_seq_lengths
                )
                fused_features, attention_map = self.gaze_fusion(
                    patch_features, cls_token, gaze_weights, gaze_features
                )
                
                # # Predict gaze from image (auxiliary task)
                # predicted_gaze = self.gaze_predictor(patch_features)  # [B, 196]
                # # Reshape to match gaze_weights [B, 196, 1]
                # target_gaze = gaze_weights.squeeze(-1)  # [B, 196]

                # Predict gaze from image (auxiliary task)
                predicted_gaze = self.gaze_predictor(patch_features)  # [B, 196]
                # For KL divergence: apply log_softmax to predictions
                predicted_gaze_log = torch.log_softmax(predicted_gaze, dim=1)  # [B, 196]
                target_gaze = gaze_weights.squeeze(-1)  # [B, 196]

                # Classify
                logits = self.classifier(fused_features)

                # # Compute losses
                # classification_loss = self.criterion(logits, labels)
                # gaze_prediction_loss = self.gaze_criterion(predicted_gaze, target_gaze)

                # Compute losses
                classification_loss = self.criterion(logits, labels)
                gaze_prediction_loss = self.gaze_criterion(predicted_gaze_log, target_gaze)

                # Total loss
                loss = (self.lambda_classification * classification_loss + 
                        self.lambda_gaze_pred * gaze_prediction_loss)
                
                # Track metrics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*correct/total:.2f}%'
                })
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs, save_dir='checkpoints'):
        """Main training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        print(f"Total epochs: {num_epochs}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Batch size: {train_loader.batch_size}")
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*80)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pth'),
                    epoch, val_acc
                )
                print(f"  ✓ New best model saved! (Val Acc: {val_acc*100:.2f}%)")
            
            # Log epoch metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc * 100,
                "val/loss": val_loss,
                "val/accuracy": val_acc * 100,
                "val/best_accuracy": self.best_val_acc * 100,
                "overfitting_gap": (train_acc - val_acc) * 100,
            })
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                    epoch, val_acc
                )
        
        # Save training history
        self.save_training_history(save_dir)

        wandb.finish()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        print(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")
    
    def save_checkpoint(self, path, epoch, val_acc):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'image_encoder_state_dict': self.image_encoder.state_dict(),
            'gaze_encoder_state_dict': self.gaze_encoder.state_dict(),
            'gaze_fusion_state_dict': self.gaze_fusion.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }, path)
    
    def save_training_history(self, save_dir):
        """Save training metrics"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


def main():
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='train',
        config=config
    )
    
    val_dataset = MIMICEyeDataset(
        root_dir='/media/16TB_Storage/kavin/dataset/processed_mimic_eye',
        split='val',
        config=config
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = C3NetTrainer(config)

    # Create unique checkpoint directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(config['training']['checkpoint_dir'], f'run_{timestamp}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=checkpoint_dir
    )

if __name__ == '__main__':
    main()