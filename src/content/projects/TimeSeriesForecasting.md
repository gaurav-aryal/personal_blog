---
title: "Advanced Time Series Forecasting System"
description: "Implementation of a sophisticated time series forecasting system using LSTM, Prophet, and ensemble methods for multi-variate prediction"
pubDate: "Feb 10 2024"
heroImage: "/post_img.webp"
---

## System Architecture

A comprehensive time series forecasting system that combines multiple models and techniques for robust predictions.

### Core Components

#### 1. Data Processing Pipeline
```python
class TimeSeriesPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_engineer = TSFeatureGenerator(
            seasonal_periods=config['seasonal_periods']
        )
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # Handle missing values
        imputed_data = self.imputer.fit_transform(data)
        
        # Generate time features
        features = self.feature_engineer.generate_features(imputed_data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        return pd.DataFrame(scaled_features, columns=features.columns)

class TSFeatureGenerator:
    def __init__(self, seasonal_periods: List[int]):
        self.seasonal_periods = seasonal_periods
        
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        
        # Add time-based features
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        
        # Add lag features
        for lag in [1, 7, 14, 30]:
            features[f'lag_{lag}'] = df.shift(lag)
            
        # Add rolling statistics
        for window in [7, 14, 30]:
            features[f'rolling_mean_{window}'] = df.rolling(window).mean()
            features[f'rolling_std_{window}'] = df.rolling(window).std()
            
        return features
```

#### 2. Model Architecture
```python
class EnsembleTimeSeriesModel:
    def __init__(self):
        self.models = {
            'lstm': self._build_lstm(),
            'prophet': Prophet(yearly_seasonality=True),
            'xgboost': XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000
            )
        }
        self.ensemble_weights = None
        
    def _build_lstm(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def train(self, X: np.ndarray, y: np.ndarray):
        predictions = {}
        
        # Train individual models
        for name, model in self.models.items():
            if name == 'prophet':
                self._train_prophet(model, X, y)
            else:
                model.fit(X, y)
            predictions[name] = self._get_predictions(model, X)
            
        # Optimize ensemble weights
        self.ensemble_weights = self._optimize_weights(predictions, y)
        
    def _optimize_weights(
        self, 
        predictions: Dict[str, np.ndarray], 
        y_true: np.ndarray
    ) -> np.ndarray:
        def objective(weights):
            weighted_pred = sum(
                w * p for w, p in zip(weights, predictions.values())
            )
            return mean_squared_error(y_true, weighted_pred)
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1)] * len(predictions)
        
        result = minimize(
            objective,
            x0=np.ones(len(predictions)) / len(predictions),
            bounds=bounds,
            constraints=constraints
        )
        return result.x
```

#### 3. Uncertainty Quantification
```python
class UncertaintyEstimator:
    def __init__(self, n_bootstrap: int = 1000):
        self.n_bootstrap = n_bootstrap
        
    def estimate_uncertainty(
        self, 
        model: EnsembleTimeSeriesModel,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        predictions = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(
                len(X), 
                size=len(X), 
                replace=True
            )
            X_boot = X[idx]
            
            # Get predictions
            pred = model.predict(X_boot)
            predictions.append(pred)
            
        predictions = np.array(predictions)
        
        # Calculate confidence intervals
        lower = np.percentile(predictions, 2.5, axis=0)
        upper = np.percentile(predictions, 97.5, axis=0)
        
        return lower, upper
```

### Model Evaluation

```python
class TimeSeriesEvaluator:
    def __init__(self):
        self.metrics = {
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'mape': mean_absolute_percentage_error,
            'rmse': lambda y, p: np.sqrt(mean_squared_error(y, p))
        }
        
    def evaluate(
        self, 
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainty: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        results = {
            name: metric(y_true, y_pred)
            for name, metric in self.metrics.items()
        }
        
        if uncertainty is not None:
            lower, upper = uncertainty
            results['coverage'] = np.mean(
                (y_true >= lower) & (y_true <= upper)
            )
            
        return results
```

### Deployment

```python
class TimeSeriesService:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.preprocessor = TimeSeriesPreprocessor(
            config={'seasonal_periods': [24, 168, 8760]}
        )
        
    async def forecast(
        self,
        historical_data: pd.DataFrame,
        horizon: int
    ) -> Dict[str, Any]:
        # Preprocess data
        processed_data = self.preprocessor.process(historical_data)
        
        # Generate forecast
        forecast = self.model.predict(processed_data, horizon)
        
        # Estimate uncertainty
        uncertainty = UncertaintyEstimator().estimate_uncertainty(
            self.model,
            processed_data
        )
        
        return {
            'forecast': forecast.tolist(),
            'lower_bound': uncertainty[0].tolist(),
            'upper_bound': uncertainty[1].tolist()
        }
```

### Usage Example

```python
# Load and prepare data
data = pd.read_csv('time_series_data.csv', parse_dates=['timestamp'])
data.set_index('timestamp', inplace=True)

# Initialize model
model = EnsembleTimeSeriesModel()

# Train model
X_train, y_train = prepare_training_data(data)
model.train(X_train, y_train)

# Make predictions with uncertainty
forecaster = TimeSeriesService('model.pkl')
forecast = await forecaster.forecast(data, horizon=30)

# Evaluate results
evaluator = TimeSeriesEvaluator()
metrics = evaluator.evaluate(
    y_test,
    forecast['forecast'],
    (forecast['lower_bound'], forecast['upper_bound'])
)
```

[View Source Code](#) | [Documentation](#) | [Contributing Guidelines](#) 