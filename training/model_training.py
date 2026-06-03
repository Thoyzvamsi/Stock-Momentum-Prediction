import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import pickle
import os
import json

class GeneralizedMLTrainer:
    """
    Train generalized model on multiple stocks combined
    """
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.training_config = {}
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_data(self, features_df, target_series):
        """
        Prepare data for training
        """
        print("Preparing data for multi-stock training...")
        
        # Combine features and target
        data = pd.concat([features_df, target_series], axis=1)
        
        # Drop NaNs
        data = data.dropna()
        print(f"  After dropping NaNs: {len(data)} samples")
        
        # Remove neutral targets
        data = data[data['target'] != 0]
        print(f"  After removing neutrals: {len(data)} samples")
        
        # Split X and y
        X = data[features_df.columns]
        y = data['target']
        
        # Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Class distribution
        class_dist = y.value_counts().to_dict()
        print(f"  Class distribution: {class_dist}")
        
        self.training_config['n_samples'] = len(X_scaled)
        self.training_config['n_features'] = X_scaled.shape[1]
        self.training_config['feature_names'] = X_scaled.columns.tolist()
        self.training_config['class_distribution'] = class_dist
        
        return X_scaled, y
    
    def walk_forward_validation(self, X, y, n_splits=5):
        """
        Walk-forward validation for time series
        Critical for multi-stock model to avoid lookahead bias
        """
        print(f"\nPerforming walk-forward validation ({n_splits} splits)...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train
            model = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.01,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.7,
                class_weight='balanced',
                verbose=-1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            scores.append(accuracy)
            
            print(f"  Fold {fold}: Accuracy = {accuracy:.3f}")
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"✓ Walk-forward average: {avg_score:.3f} (+/- {std_score:.3f})")
        
        self.training_config['wfv_mean'] = float(avg_score)
        self.training_config['wfv_std'] = float(std_score)
        
        return scores
    
    def train_final_model(self, X_train, y_train):
        """Train final model on all training data"""
        print("\nTraining final generalized model...")
        
        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.7,
            class_weight='balanced',
            verbose=-1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        print("✓ Model trained")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        print("\nModel evaluation on test set:")
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"  Accuracy:  {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist()
        }
        
        self.training_config['test_accuracy'] = float(accuracy)
        self.training_config['test_precision'] = float(precision)
        self.training_config['test_recall'] = float(recall)
        self.training_config['test_f1'] = float(f1)
        
        return metrics
    
    def feature_importance(self, top_n=15):
        """Get feature importance"""
        importance_df = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        self.training_config['top_features'] = importance_df.head(top_n)[['feature', 'importance']].to_dict(orient='list')
        
        return importance_df
    
    def save_model(self, name='generalized_momentum'):
        """Save model, scaler, and config"""
        model_path = os.path.join(self.model_dir, f'{name}_model.pkl')
        scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
        config_path = os.path.join(self.model_dir, f'{name}_config.json')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(config_path, 'w') as f:
            json.dump(self.training_config, f, indent=2)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Scaler saved to {scaler_path}")
        print(f"✓ Config saved to {config_path}")
    
    def load_model(self, name='generalized_momentum'):
        """Load saved model, scaler, and config"""
        model_path = os.path.join(self.model_dir, f'{name}_model.pkl')
        scaler_path = os.path.join(self.model_dir, f'{name}_scaler.pkl')
        config_path = os.path.join(self.model_dir, f'{name}_config.json')
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(config_path, 'r') as f:
                self.training_config = json.load(f)
            
            print(f"✓ Model loaded from {model_path}")
            print(f"✓ Scaler loaded from {scaler_path}")
            print(f"✓ Config loaded from {config_path}")
            
            return self.model, self.scaler, self.training_config
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return None, None, None
    
    def predict(self, X_scaled):
        """Make predictions"""
        if self.model is None:
            print("✗ Model not loaded")
            return None, None
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities


class DataSplitter:
    """Time-aware data splitter"""
    
    @staticmethod
    def train_test_split_timeseries(X, y, test_size=0.2):
        """Split preserving time order"""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data.pipeline import MultiStockDataPipeline
    from features.engineering import GeneralizedFeatureEngineer, GeneralizedTargetEngineer
    
    # Load training data
    pipeline = MultiStockDataPipeline()
    df = pipeline.load_training_data()
    
    if df is not None:
        # Create features
        fe = GeneralizedFeatureEngineer(df)
        features = fe.create_all_features()
        
        # Create target
        target = GeneralizedTargetEngineer.create_forward_return_target(df)
        
        # Prepare
        trainer = GeneralizedMLTrainer()
        X, y = trainer.prepare_data(features, target)
        
        # Walk-forward validation
        trainer.walk_forward_validation(X, y, n_splits=5)
        
        # Split and train
        X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(X, y)
        trainer.train_final_model(X_train, y_train)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        
        # Feature importance
        trainer.feature_importance(top_n=15)
        
        # Save
        trainer.save_model('generalized_momentum')
