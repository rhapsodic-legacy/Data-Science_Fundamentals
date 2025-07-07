# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_dataset():
    """
    Load the credit card fraud dataset from Kaggle.
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    print("Downloading dataset from Kaggle...")
    # Download the dataset using kagglehub
    path = kagglehub.dataset_download("dhanushnarayananr/credit-card-fraud")
    print(f"Dataset downloaded to: {path}")
    
    # Find and load the CSV file
    csv_files = glob.glob(path + '/*.csv')
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the dataset directory")
    
    df = pd.read_csv(csv_files[0])
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def explore_dataset(df):
    """
    Perform basic exploratory data analysis on the dataset.
    
    Args:
        df (pandas.DataFrame): The dataset to explore
    """
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Check class distribution
    fraud_counts = df['fraud'].value_counts()
    print(f"\nClass distribution:")
    print(f"Non-fraud transactions: {fraud_counts[0]} ({fraud_counts[0]/len(df)*100:.2f}%)")
    print(f"Fraud transactions: {fraud_counts[1]} ({fraud_counts[1]/len(df)*100:.2f}%)")
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nDataset info:")
    print(df.info())

def feature_engineering(df):
    """
    Create new features to improve model performance.
    
    Feature engineering is crucial for anomaly detection as it helps the model
    identify more subtle patterns that distinguish fraudulent from normal transactions.
    
    Args:
        df (pandas.DataFrame): Original dataset
        
    Returns:
        pandas.DataFrame: Dataset with engineered features
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50)
    
    df_engineered = df.copy()
    
    # Feature 1: Combined distance and price ratio
    # Rationale: Fraudulent transactions might occur far from home with unusual amounts
    df_engineered['distance_ratio'] = (
        df_engineered['distance_from_home'] * 
        df_engineered['ratio_to_median_purchase_price']
    )
    
    # Feature 2: Distance from last transaction combined with price ratio
    # Rationale: Quick successive transactions in different locations with unusual amounts
    df_engineered['distance_last_transaction_ratio'] = (
        df_engineered['distance_from_last_transaction'] * 
        df_engineered['ratio_to_median_purchase_price']
    )
    
    # Feature 3: Average ratio per retailer
    # Rationale: Each retailer might have different typical transaction patterns
    avg_ratio_retailer = df_engineered.groupby('repeat_retailer')['ratio_to_median_purchase_price'].transform('mean')
    df_engineered['avg_ratio_retailer'] = avg_ratio_retailer
    
    # Feature 4: Standard deviation of ratio per retailer
    # Rationale: Measure of transaction variability per retailer
    std_ratio_retailer = df_engineered.groupby('repeat_retailer')['ratio_to_median_purchase_price'].transform('std')
    df_engineered['std_ratio_retailer'] = std_ratio_retailer.fillna(0)  # Fill NaN for single-transaction retailers
    
    # Feature 5: Transaction frequency per retailer
    # Rationale: Popular retailers vs. rare ones might have different fraud patterns
    transaction_frequency = df_engineered['repeat_retailer'].value_counts().to_dict()
    df_engineered['transaction_frequency'] = df_engineered['repeat_retailer'].map(transaction_frequency)
    
    print(f"Original features: {df.shape[1]}")
    print(f"Features after engineering: {df_engineered.shape[1]}")
    print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")
    
    return df_engineered

def prepare_data(df):
    """
    Prepare data for training by separating features and target, and splitting into train/test sets.
    
    Args:
        df (pandas.DataFrame): The engineered dataset
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    # Separate features and target
    X = df.drop('fraud', axis=1)
    y = df['fraud']
    
    # Split into training and testing sets
    # Stratify ensures both sets have similar fraud/non-fraud ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for better neural network performance
    # Autoencoders are sensitive to feature scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Training set fraud ratio: {y_train.mean():.4f}")
    print(f"Test set fraud ratio: {y_test.mean():.4f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_autoencoder(input_dim, encoding_dim=4, learning_rate=0.001):
    """
    Create and compile an autoencoder model.
    
    An autoencoder consists of:
    1. Encoder: Compresses input to lower-dimensional representation
    2. Decoder: Reconstructs input from compressed representation
    
    For anomaly detection, we train on normal data. Anomalies should have
    higher reconstruction errors since they differ from normal patterns.
    
    Args:
        input_dim (int): Number of input features
        encoding_dim (int): Size of the compressed representation (bottleneck)
        learning_rate (float): Learning rate for optimization
        
    Returns:
        tensorflow.keras.Model: Compiled autoencoder model
    """
    # Input layer
    input_layer = Input(shape=(input_dim,), name='input')
    
    # Encoder: Compress input to lower dimension
    encoded = Dense(
        encoding_dim, 
        activation='relu', 
        name='encoder'
    )(input_layer)
    
    # Decoder: Reconstruct input from compressed representation
    decoded = Dense(
        input_dim, 
        activation='sigmoid',  # Sigmoid ensures output is between 0 and 1
        name='decoder'
    )(encoded)
    
    # Create the autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
    
    # Compile with Adam optimizer and MSE loss
    # MSE measures reconstruction quality
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']  # Mean Absolute Error for additional monitoring
    )
    
    return autoencoder

def train_autoencoder(autoencoder, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the autoencoder on non-fraudulent transactions only.
    
    Key insight: We only train on normal transactions so the model learns
    to reconstruct normal patterns well. Fraudulent transactions should
    have higher reconstruction errors.
    
    Args:
        autoencoder: The autoencoder model
        X_train: Training features
        y_train: Training labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tensorflow.keras.callbacks.History: Training history
    """
    print("\n" + "="*50)
    print("TRAINING AUTOENCODER")
    print("="*50)
    
    # Filter to only non-fraudulent transactions
    normal_transactions = X_train[y_train == 0]
    print(f"Training on {len(normal_transactions)} normal transactions")
    print(f"Excluded {len(X_train) - len(normal_transactions)} fraudulent transactions")
    
    # Train autoencoder to reconstruct normal transactions
    # Input and output are the same (unsupervised learning)
    history = autoencoder.fit(
        normal_transactions, 
        normal_transactions,  # Target is same as input
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # Use 10% for validation
        verbose=1,
        shuffle=True
    )
    
    return history

def evaluate_model(autoencoder, X_test, y_test):
    """
    Evaluate the autoencoder's performance for fraud detection.
    
    We use reconstruction error as an anomaly score:
    - Low error: Normal transaction (good reconstruction)
    - High error: Potential fraud (poor reconstruction)
    
    Args:
        autoencoder: Trained autoencoder model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        tuple: (reconstruction_errors, auc_score)
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Get reconstructions for test set
    reconstructions = autoencoder.predict(X_test, verbose=0)
    
    # Calculate reconstruction error (MSE) for each transaction
    reconstruction_errors = np.mean((X_test - reconstructions) ** 2, axis=1)
    
    # Calculate AUC score
    # Higher reconstruction error should correlate with fraud
    auc_score = roc_auc_score(y_test, reconstruction_errors)
    
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Mean reconstruction error (normal): {reconstruction_errors[y_test == 0].mean():.6f}")
    print(f"Mean reconstruction error (fraud): {reconstruction_errors[y_test == 1].mean():.6f}")
    
    return reconstruction_errors, auc_score

def plot_roc_curve(y_test, reconstruction_errors):
    """
    Plot ROC curve to visualize model performance.
    
    ROC curve shows the trade-off between:
    - True Positive Rate (sensitivity): Fraud correctly identified
    - False Positive Rate: Normal transactions incorrectly flagged
    
    Args:
        y_test: True labels
        reconstruction_errors: Anomaly scores (reconstruction errors)
    """
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(y_test, reconstruction_errors)
    auc_value = auc(fpr, tpr)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_value:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Fraud Detection Performance')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def hyperparameter_optimization(X_train, X_test, y_train, y_test):
    """
    Perform grid search to find optimal hyperparameters.
    
    This is computationally expensive but important for maximizing performance.
    In practice, you might use more sophisticated optimization techniques.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        
    Returns:
        dict: Best hyperparameters found
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    # Define hyperparameter search space
    epochs_list = [50, 100]  # Reduced for faster execution
    batch_size_list = [32, 64]
    learning_rate_list = [0.001, 0.01]
    encoding_dim_list = [4, 8]
    
    best_auc = 0
    best_params = None
    results = []
    
    total_combinations = len(epochs_list) * len(batch_size_list) * len(learning_rate_list) * len(encoding_dim_list)
    combination = 0
    
    print(f"Testing {total_combinations} combinations...")
    
    for epochs in epochs_list:
        for batch_size in batch_size_list:
            for learning_rate in learning_rate_list:
                for encoding_dim in encoding_dim_list:
                    combination += 1
                    print(f"Combination {combination}/{total_combinations}: "
                          f"epochs={epochs}, batch_size={batch_size}, "
                          f"lr={learning_rate}, encoding_dim={encoding_dim}")
                    
                    # Create and train model
                    autoencoder = create_autoencoder(
                        X_train.shape[1], 
                        encoding_dim=encoding_dim,
                        learning_rate=learning_rate
                    )
                    
                    # Train on normal transactions only
                    normal_transactions = X_train[y_train == 0]
                    autoencoder.fit(
                        normal_transactions, 
                        normal_transactions,
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=0
                    )
                    
                    # Evaluate
                    predictions = autoencoder.predict(X_test, verbose=0)
                    mse = np.mean((X_test - predictions) ** 2, axis=1)
                    auc_score = roc_auc_score(y_test, mse)
                    
                    # Store results
                    result = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'encoding_dim': encoding_dim,
                        'auc': auc_score
                    }
                    results.append(result)
                    
                    # Update best parameters
                    if auc_score > best_auc:
                        best_auc = auc_score
                        best_params = result.copy()
                    
                    print(f"  AUC: {auc_score:.4f}")
    
    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best AUC: {best_auc:.4f}")
    
    return best_params, results

def main():
    """
    Main function to orchestrate the fraud detection pipeline.
    """
    print("Credit Card Fraud Detection using Autoencoders")
    print("=" * 60)
    
    try:
        # 1. Load and explore dataset
        df = load_dataset()
        explore_dataset(df)
        
        # 2. Feature engineering
        df_engineered = feature_engineering(df)
        
        # 3. Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(df_engineered)
        
        # 4. Create and train autoencoder
        autoencoder = create_autoencoder(
            input_dim=X_train.shape[1],
            encoding_dim=4,
            learning_rate=0.001
        )
        
        print("\nAutoencoder Architecture:")
        autoencoder.summary()
        
        # Train the model
        history = train_autoencoder(
            autoencoder, X_train, y_train, 
            epochs=50, batch_size=32
        )
        
        # 5. Evaluate model
        reconstruction_errors, auc_score = evaluate_model(autoencoder, X_test, y_test)
        
        # 6. Visualize results
        plot_roc_curve(y_test, reconstruction_errors)
        
        # 7. Optional: Hyperparameter optimization (commented out for faster execution)
        # print("\nStarting hyperparameter optimization...")
        # best_params, results = hyperparameter_optimization(X_train, X_test, y_train, y_test)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Final AUC Score: {auc_score:.4f}")
        
        # Interpretation guide
        print("\nInterpretation:")
        if auc_score > 0.9:
            print("- Excellent performance! The model effectively distinguishes fraud.")
        elif auc_score > 0.8:
            print("- Good performance. The model shows strong fraud detection capability.")
        elif auc_score > 0.7:
            print("- Fair performance. Consider more feature engineering or model tuning.")
        else:
            print("- Poor performance. The model needs significant improvement.")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please ensure you have the required packages installed:")
        print("pip install tensorflow pandas scikit-learn matplotlib kagglehub")

if __name__ == "__main__":
    main()