import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import pickle
import os
import json
from training.config import (
    FINAL_MODEL_PARAMS,
    MODEL_NAME,
    MODELS_DIR,
    WALK_FORWARD_MODEL_PARAMS,
)

class GeneralizedMLTrainer:
    """
    Train generalized model on multiple stocks combined
    """
    
    def __init__(self, model_dir=MODELS_DIR):
        self.model_dir = str(model_dir)
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
        
        # Class distribution
        class_dist = y.value_counts().to_dict()
        print(f"  Class distribution: {class_dist}")
        
        self.training_config['n_samples'] = len(X)
        self.training_config['n_features'] = X.shape[1]
        self.training_config['feature_names'] = X.columns.tolist()
        self.training_config['class_distribution'] = class_dist
        
        return X, y
    
    def walk_forward_validation(
        self, X, y, n_splits=5, timestamps=None, gap=0, model_params=None
    ):
        """
        Walk-forward validation for time series
        Critical for multi-stock model to avoid lookahead bias
        """
        print(f"\nPerforming walk-forward validation ({n_splits} splits)...")
        
        if timestamps is None:
            split_source = X
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
            time_values = None
        else:
            time_values = pd.Series(
                pd.to_datetime(timestamps), index=X.index
            )
            unique_times = pd.Index(time_values.unique()).sort_values()
            split_source = unique_times
            tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(
            tscv.split(split_source), 1
        ):
            if time_values is None:
                train_positions, val_positions = train_idx, val_idx
            else:
                train_times = split_source[train_idx]
                val_times = split_source[val_idx]
                train_positions = np.flatnonzero(time_values.isin(train_times))
                val_positions = np.flatnonzero(time_values.isin(val_times))

            X_train = X.iloc[train_positions]
            X_val = X.iloc[val_positions]
            y_train = y.iloc[train_positions]
            y_val = y.iloc[val_positions]
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val), columns=X.columns, index=X_val.index
            )

            # Train
            model = lgb.LGBMClassifier(
                **(model_params or WALK_FORWARD_MODEL_PARAMS)
            )
            print(
                f"  Fold {fold}: training {model.n_estimators} rounds "
                f"({len(X_train):,} train / {len(X_val):,} validation rows)"
            )
            model.fit(
                X_train_scaled,
                y_train,
                eval_set=[
                    (X_train_scaled, y_train),
                    (X_val_scaled, y_val),
                ],
                eval_names=["train", "valid"],
                eval_metric="binary_logloss",
                callbacks=[lgb.log_evaluation(period=50)],
            )
            
            # Evaluate
            y_pred = model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            scores.append(accuracy)
            
            print(f"  Fold {fold}: Accuracy = {accuracy:.3f}")
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"Walk-forward average: {avg_score:.3f} (+/- {std_score:.3f})")
        
        self.training_config['wfv_mean'] = float(avg_score)
        self.training_config['wfv_std'] = float(std_score)
        
        return scores
    
    def train_final_model(self, X_train, y_train, model_params=None):
        """Train final model on all training data"""
        print("\nTraining final generalized model...")
        
        self.model = lgb.LGBMClassifier(**(model_params or FINAL_MODEL_PARAMS))
        
        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        self.model.fit(X_train_scaled, y_train)
        print("Model trained")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        print("\nModel evaluation on test set:")
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)
        
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
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Config saved to {config_path}")
    
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
            
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
            print(f"Config loaded from {config_path}")
            
            return self.model, self.scaler, self.training_config
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None, None, None
    
    def predict(self, X_scaled):
        """Make predictions"""
        if self.model is None:
            print("Model not loaded")
            return None, None
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities


class DataSplitter:
    """Time-aware data splitter"""
    
    @staticmethod
    def train_test_split_timeseries(
        X, y, test_size=0.2, timestamps=None, gap=0
    ):
        """Split chronologically, keeping each timestamp together and embargoing the boundary."""
        if timestamps is None:
            split_idx = int(len(X) * (1 - test_size))
            train_end = split_idx - gap
            if train_end <= 0:
                raise ValueError("The requested embargo leaves no training rows.")
            train_positions = np.arange(train_end)
            test_positions = np.arange(split_idx, len(X))
        else:
            time_values = pd.Series(pd.to_datetime(timestamps), index=X.index)
            unique_times = pd.Index(time_values.unique()).sort_values()
            split_idx = int(len(unique_times) * (1 - test_size))
            train_end = split_idx - gap
            if train_end <= 0 or split_idx >= len(unique_times):
                raise ValueError("The requested split and embargo leave an empty partition.")
            train_times = unique_times[:train_end]
            test_times = unique_times[split_idx:]
            train_positions = np.flatnonzero(time_values.isin(train_times))
            test_positions = np.flatnonzero(time_values.isin(test_times))

        X_train = X.iloc[train_positions]
        X_test = X.iloc[test_positions]
        y_train = y.iloc[train_positions]
        y_test = y.iloc[test_positions]
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from training.config import FORWARD_BARS, RAW_DATA_FILE
    from training.features.engineering import GeneralizedFeatureEngineer, GeneralizedTargetEngineer
    
    df = pd.read_csv(RAW_DATA_FILE)
    if 'Datetime' not in df.columns:
        raise ValueError("Training data must include a Datetime column.")
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values(['Datetime', 'Ticker'], kind='stable').reset_index(drop=True)

    features = GeneralizedFeatureEngineer(df).create_all_features()
    features_clean = features.dropna()
    df_clean = df.loc[features_clean.index]
    target = GeneralizedTargetEngineer.create_forward_return_target(
        df_clean, forward_bars=FORWARD_BARS
    )

    trainer = GeneralizedMLTrainer()
    X, y = trainer.prepare_data(features_clean, target)
    timestamps = df_clean.loc[X.index, 'Datetime']
    trainer.walk_forward_validation(
        X, y, n_splits=5, timestamps=timestamps, gap=FORWARD_BARS
    )

    X_train, X_test, y_train, y_test = DataSplitter.train_test_split_timeseries(
        X, y, timestamps=timestamps, gap=FORWARD_BARS
    )
    trainer.train_final_model(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.feature_importance(top_n=15)
    trainer.save_model(MODEL_NAME)
