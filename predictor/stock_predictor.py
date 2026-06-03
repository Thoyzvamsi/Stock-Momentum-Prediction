import pandas as pd
import numpy as np
from features.engineering import GeneralizedFeatureEngineer
from training.model_training import GeneralizedMLTrainer

class StockPredictor:
    """
    Single stock predictor using trained generalized model
    """
    
    def __init__(self, model_name='generalized_momentum', models_dir='models'):
        """
        Args:
            model_name: Name of trained model
            models_dir: Directory containing models
        """
        self.trainer = GeneralizedMLTrainer(model_dir=models_dir)
        self.model, self.scaler, self.config = self.trainer.load_model(model_name)
        
        if self.model is None:
            raise ValueError(f"Could not load model: {model_name}")
        
        self.feature_names = self.config.get('feature_names', [])
    
    def predict_stock(self, df, return_features=False):
        """
        Generate predictions for a stock
        
        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume, Ticker]
            return_features: If True, return features along with predictions
        
        Returns:
            Dict with predictions, probabilities, signals
        """
        
        # Create features using same pipeline
        fe = GeneralizedFeatureEngineer(df)
        features = fe.create_all_features()
        
        # Drop NaNs
        features_clean = features.dropna()
        df_clean = df.loc[features_clean.index]
        
        if len(features_clean) == 0:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features_clean)
        features_scaled = pd.DataFrame(
            features_scaled, 
            columns=features_clean.columns, 
            index=features_clean.index
        )
        
        # Predict
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Get max probability (confidence)
        confidence = probabilities.max(axis=1)
        
        # Expected return (if available from historical data)
        # This would be from a separate calibration
        expected_returns = self._estimate_expected_return(
            predictions, confidence, probabilities
        )
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence': confidence,
            'expected_returns': expected_returns,
            'timestamps': df_clean.index,
            'prices': df_clean['Close'].values,
            'features': features_clean if return_features else None
        }
        
        return results
    
    def predict_latest(self, df):
        """
        Get prediction for the latest bar only
        
        Args:
            df: DataFrame with latest data
        
        Returns:
            Dict with latest signal
        """
        results = self.predict_stock(df)
        
        if results is None:
            return None
        
        latest_idx = -1
        
        latest_result = {
            'timestamp': results['timestamps'][latest_idx],
            'price': results['prices'][latest_idx],
            'prediction': results['predictions'][latest_idx],
            'signal': self._prediction_to_signal(results['predictions'][latest_idx]),
            'probability': results['probabilities'][latest_idx],
            'confidence': results['confidence'][latest_idx],
            'expected_return': results['expected_returns'][latest_idx]
        }
        
        return latest_result
    
    def _prediction_to_signal(self, prediction):
        """Convert prediction to signal"""
        if prediction == 1:
            return 'BUY'
        elif prediction == -1:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _estimate_expected_return(self, predictions, confidence, probabilities):
        """
        Estimate expected return based on prediction and confidence
        This is approximate - in production you'd calibrate this properly
        """
        # Base expected returns by prediction
        base_returns = np.array([
            0.005 if p == 1 else (-0.003 if p == -1 else 0.0001)
            for p in predictions
        ])
        
        # Adjust by confidence
        confidence_adjusted = base_returns * (confidence / 0.5)  # Normalized by 50%
        
        return confidence_adjusted
    
    def get_model_info(self):
        """Get information about the trained model"""
        info = {
            'n_samples': self.config.get('n_samples'),
            'n_features': self.config.get('n_features'),
            'feature_names': self.feature_names,
            'wfv_mean': self.config.get('wfv_mean'),
            'wfv_std': self.config.get('wfv_std'),
            'test_accuracy': self.config.get('test_accuracy'),
            'test_precision': self.config.get('test_precision'),
            'test_recall': self.config.get('test_recall'),
            'test_f1': self.config.get('test_f1'),
            'top_features': self.config.get('top_features')
        }
        return info


class SignalGenerator:
    """
    Generate trading signals from predictions
    """
    
    @staticmethod
    def generate_signal(prediction, confidence, min_confidence=0.65):
        """
        Generate actionable signal
        
        Args:
            prediction: Model prediction (-1, 0, 1)
            confidence: Confidence score (0-1)
            min_confidence: Minimum confidence threshold
        
        Returns:
            Signal dict
        """
        
        if confidence < min_confidence:
            signal = 'HOLD'
            strength = 'WEAK'
        else:
            if prediction == 1:
                signal = 'BUY'
                strength = 'STRONG' if confidence > 0.75 else 'MODERATE'
            elif prediction == -1:
                signal = 'SELL'
                strength = 'STRONG' if confidence > 0.75 else 'MODERATE'
            else:
                signal = 'HOLD'
                strength = 'NEUTRAL'
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'action': _signal_to_action(signal, strength)
        }
    
    @staticmethod
    def batch_generate_signals(predictions, confidences, min_confidence=0.65):
        """Generate signals for multiple predictions"""
        signals = []
        for pred, conf in zip(predictions, confidences):
            signal = SignalGenerator.generate_signal(pred, conf, min_confidence)
            signals.append(signal)
        
        return signals


def _signal_to_action(signal, strength):
    """Convert signal to action"""
    if signal == 'BUY':
        return 'ENTER_LONG' if strength == 'STRONG' else 'CONSIDER_LONG'
    elif signal == 'SELL':
        return 'ENTER_SHORT' if strength == 'STRONG' else 'CONSIDER_SHORT'
    else:
        return 'NO_ACTION'


if __name__ == "__main__":
    from data.pipeline import MultiStockDataPipeline
    
    # Load predictor
    predictor = StockPredictor('generalized_momentum')
    
    # Get model info
    info = predictor.get_model_info()
    print("Model Information:")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Test Accuracy: {info['test_accuracy']:.3f}")
    
    # Fetch live data and predict
    pipeline = MultiStockDataPipeline()
    df = pipeline.fetch_live_data('AAPL', period='3mo')
    
    if df is not None:
        # Get full results
        results = predictor.predict_stock(df)
        
        # Get latest prediction
        latest = predictor.predict_latest(df)
        
        print(f"\nLatest Prediction for AAPL:")
        print(f"  Timestamp: {latest['timestamp']}")
        print(f"  Price: ${latest['price']:.2f}")
        print(f"  Signal: {latest['signal']}")
        print(f"  Confidence: {latest['confidence']:.1%}")
        print(f"  Expected Return: {latest['expected_return']:.2%}")
        
        # Generate signal
        signal = SignalGenerator.generate_signal(
            latest['prediction'],
            latest['confidence'],
            min_confidence=0.65
        )
        
        print(f"\nTrading Signal:")
        print(f"  Signal: {signal['signal']}")
        print(f"  Strength: {signal['strength']}")
        print(f"  Action: {signal['action']}")
