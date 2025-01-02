## TEST SCRIPT WITH SAVED MODEL
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import Dataset, DataLoader

class EVBatteryRNN(nn.Module):
    def _init_(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(EVBatteryRNN, self)._init_()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class EVBatteryDataset(Dataset):
    def _init_(self, X):
        self.X = torch.FloatTensor(X)
    
    def _len_(self):
        return len(self.X)
    
    def _getitem_(self, idx):
        return self.X[idx]

class StandalonePredictor:
    def _init_(self, model_dir):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model and parameters
        checkpoint = torch.load(f"{model_dir}/model.pth", map_location=self.device)
        model_params = checkpoint['model_params']
        
        self.model = EVBatteryRNN(
            input_size=model_params['input_size'],
            hidden_size=model_params['hidden_size'],
            num_layers=model_params['num_layers'],
            output_size=model_params['output_size']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.scalers = joblib.load(f"{model_dir}/scalers.joblib")
        self.label_encoders = joblib.load(f"{model_dir}/encoders.joblib")
    
    def prepare_data(self, df):
        """Prepare input data for prediction"""
        print("\nOriginal columns:", df.columns.tolist())
        
        df_processed = df.copy()
        
        # Drop unnamed columns if they exist
        unnamed_cols = [col for col in df_processed.columns if 'Unnamed' in col]
        if unnamed_cols:
            df_processed = df_processed.drop(columns=unnamed_cols)
        
        print("\nColumns after dropping unnamed:", df_processed.columns.tolist())
        
        # Convert timestamp to numerical features
        df_processed['Timestamp'] = pd.to_datetime(df_processed['Timestamp'])
        df_processed['Hour'] = df_processed['Timestamp'].dt.hour
        df_processed['Day'] = df_processed['Timestamp'].dt.day
        df_processed['Month'] = df_processed['Timestamp'].dt.month
        
        # Handle categorical variables
        categorical_cols = ['Road Condition', 'Traffic Density', 'Weather Condition']
        for col in categorical_cols:
            if col not in df_processed.columns:
                raise KeyError(f"Required column '{col}' not found in input data")
            le = self.label_encoders[col]
            df_processed[col] = df_processed[col].map(
                lambda x: x if x in le.classes_ else 'unknown'
            )
            df_processed[col] = le.transform(df_processed[col])
        
        # Prepare feature columns
        feature_cols = [
            'Hour', 'Day', 'Month',
            'Acceleration (m/s^2)', 'Velocity (km/h)',
            'Road Condition', 'Traffic Density', 'Weather Condition',
            'Battery SoC (%)', 'Battery Temperature (Â°C)',
            'Voltage (V)', 'Current (A)'
        ]
        
        # Check if all required columns are present
        missing_cols = [col for col in feature_cols if col not in df_processed.columns]
        if missing_cols:
            print("\nMissing columns:", missing_cols)
            print("\nAvailable columns:", df_processed.columns.tolist())
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        X = df_processed[feature_cols].values
        X = self.scalers['features'].transform(X)
        return X
    
    def prepare_sequences(self, X, sequence_length=10):
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        return np.array(sequences)
    
    def predict(self, df):
        print("Preprocessing data...")
        try:
            X = self.prepare_data(df)
            X_seq = self.prepare_sequences(X)
            
            dataset = EVBatteryDataset(X_seq)
            dataloader = DataLoader(dataset, batch_size=32)
            
            print("Making predictions...")
            predictions = []
            
            with torch.no_grad():
                for batch_X in dataloader:
                    batch_X = batch_X.to(self.device)
                    outputs = self.model(batch_X)
                    predictions.append(outputs.cpu().numpy())
            
            predictions = np.vstack(predictions)
            predictions = self.scalers['target'].inverse_transform(predictions)
            
            return pd.DataFrame(
                predictions,
                columns=['Runtime Hours Remaining', 'Distance Remaining (km)']
            )
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
            raise

if _name_ == "_main_":
    try:
        # Load the saved model
        model_dir = ".\model.pth"
        predictor = StandalonePredictor(model_dir)
        
        # Load and preprocess test data
        print("\nLoading test data...")
        test_data = pd.read_csv('test.csv')
        print("\nTest data shape:", test_data.shape)
        print("\nTest data columns:", test_data.columns.tolist())
        
        # Make predictions
        predictions = predictor.predict(test_data)
        print("\nPredictions:")
        print(predictions)
        
        # Save predictions to CSV
        # Add original timestamp column to predictions if available
        if 'Timestamp' in test_data.columns:
            predictions_with_time = pd.concat([
                test_data['Timestamp'].iloc[9:].reset_index(drop=True),  # Adjust for sequence length
                predictions
            ], axis=1)
        else:
            predictions_with_time = predictions
        
        # Save to CSV
        output_file = 'predictions.csv'
        predictions_with_time.to_csv(output_file, index=False)
        print(f"\nPredictions saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
