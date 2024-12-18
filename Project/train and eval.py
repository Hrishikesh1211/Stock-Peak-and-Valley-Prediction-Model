
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

USE_CLASS_WEIGHT = True #use class waiting
USE_BALANCED = True     #use sklearn for class waiting (false = manual version)
MANUAL_CAP = 5.5        #maximum weight assigned to each label
LEARNING_RATE = 0.0005
EPOCHS = 3
BATCH_SIZE = 64
SEQUENCE_LENGTH = 390

#file paths (updated to include the 'data' directory)
seq_train_file = 'data/sequential_features_normalized_window12.csv'
seq_val_file = 'data/sequential_features_normalized_window12_val.csv'

static_train_file = 'data/static_features_normalized_window12.csv'
static_val_file = 'data/static_features_normalized_window12_val.csv'

label_train_file = 'data/labels_interval_5_window_12_train.csv'
label_val_file = 'data/labels_interval_5_window_12_val.csv'

#original labels: -1 (valley), 0 (neither), 1 (peak)
#mapped labels: 2 (valley), 0 (neither), 1 (peak)
def map_labels(labels):
    label_mapping = {-1: 2, 0: 0, 1: 1}
    return np.vectorize(label_mapping.get)(labels)

#load training data
seq_train_df = pd.read_csv(seq_train_file, parse_dates=['Timestamp'])
static_train_df = pd.read_csv(static_train_file, parse_dates=['Timestamp'])
label_train_df = pd.read_csv(label_train_file, parse_dates=['Timestamp'])

#load validation data
seq_val_df = pd.read_csv(seq_val_file, parse_dates=['Timestamp'])
static_val_df = pd.read_csv(static_val_file, parse_dates=['Timestamp'])
label_val_df = pd.read_csv(label_val_file, parse_dates=['Timestamp'])

#verify all tables are chronologically aligned
def verify_alignment(*dfs):
    for i in range(1, len(dfs)):
        if not dfs[0]['Timestamp'].equals(dfs[i]['Timestamp']):
            raise ValueError(f"Timestamps are not aligned across DataFrames at position {i}.")

verify_alignment(seq_train_df, static_train_df, label_train_df)
verify_alignment(seq_val_df, static_val_df, label_val_df)

#feature columns
seq_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD']
static_features = [
    'Inflation', 'Inflation_RateChange', 'InterestRate', 'InterestRate_RateChange',
    'Sentiment_Avg', 'Sentiment_SD', 'HourOfDay',
    'DayOfWeek_Monday', 'DayOfWeek_Tuesday', 'DayOfWeek_Wednesday',
    'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfMonth'
]

#convert dataframe columns to numpy arrays
seq_train_data = seq_train_df[seq_features].values
static_train_data = static_train_df[static_features].values.astype(np.float32)
Y_train_original = label_train_df['Class'].values
Y_train = map_labels(Y_train_original)

seq_val_data = seq_val_df[seq_features].values
static_val_data = static_val_df[static_features].values.astype(np.float32)
Y_val_original = label_val_df['Class'].values
Y_val = map_labels(Y_val_original)

def create_sequences(seq_data, static_data, labels, sequence_length):
    """
    - seq_data (np.array): Sequential features.
    - static_data (np.array): Static features.
    - labels (np.array): Labels corresponding to each interval.
    - sequence_length (int): Number of past intervals to include in each sequence.

    returns:
    - X_seq (np.array): Sequences of sequential features.
    - X_static (np.array): Corresponding static features.
    - Y (np.array): Labels for each sequence.
    """
    X_seq = []
    X_static = []
    Y = []
    num_samples = len(seq_data)

    for i in range(sequence_length, num_samples):
        X_seq.append(seq_data[i-sequence_length:i, :])
        X_static.append(static_data[i])
        Y.append(labels[i])

    return np.array(X_seq), np.array(X_static), np.array(Y)

#create sequences for training
X_seq_train, X_static_train, Y_train = create_sequences(seq_train_data, static_train_data, Y_train, SEQUENCE_LENGTH)

#create sequences for validation
X_seq_val, X_static_val, Y_val = create_sequences(seq_val_data, static_val_data, Y_val, SEQUENCE_LENGTH)

print(f"Training Sequences: {X_seq_train.shape}")
print(f"Validation Sequences: {X_seq_val.shape}")

class_weight_dict = None

if USE_CLASS_WEIGHT:
    unique_classes = np.unique(Y_train)
    if USE_BALANCED:
        #use sklearn for balancing
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=Y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}
    else:
        #manual balancing: total_samples / (num_classes * class_count) (not necessary anymore)
        class_counts = np.bincount(Y_train)
        total_samples = len(Y_train)
        num_classes = len(unique_classes)
        class_weight_dict = {cls: total_samples / (num_classes * count) for cls, count in zip(unique_classes, class_counts)}

    #manual cap for limiting weight of peak/valley
    if MANUAL_CAP is not None:
        class_weight_dict = {c: min(w, MANUAL_CAP) for c, w in class_weight_dict.items()}

    print("Using class weights:", class_weight_dict)
else:
    print("Not using any class weights.")

def build_model(sequence_length, seq_feature_dim, static_feature_dim, learning_rate=0.0005):
    """
    - sequence_length (int): Length of input sequence
    - seq_feature_dim (int): Number of sequential features
    - static_feature_dim (int): Number of static features
    - learning_rate (float): Optimizer learning rate

    returns:
    - model (model)
    """
    #sequential input
    seq_input = Input(shape=(sequence_length, seq_feature_dim), name='sequence_input')
    lstm_out = LSTM(units=128, dropout=0.2, return_sequences=False, activation='tanh')(seq_input)

    #static input
    static_input = Input(shape=(static_feature_dim,), name='static_input')
    static_dense = Dense(32, activation='relu')(static_input)
    static_dense = Dropout(0.3)(static_dense)

    #combination layer
    combined = Concatenate()([lstm_out, static_dense])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.3)(combined)

    # Output layer
    output = Dense(3, activation='softmax', name='class_output')(combined)

    model = Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

seq_feature_dim = X_seq_train.shape[2]
static_feature_dim = X_static_train.shape[1]

model = build_model(SEQUENCE_LENGTH, seq_feature_dim, static_feature_dim, LEARNING_RATE)
model.summary()

history = model.fit(
    [X_seq_train, X_static_train],
    Y_train,
    validation_data=([X_seq_val, X_static_val], Y_val),
    class_weight=class_weight_dict,  # This can be None if not using weights
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

#evaluation portion
evaluation = model.evaluate([X_seq_val, X_static_val], Y_val, verbose=0)

print("\nModel Evaluation on Full Validation Set:")
for name, value in zip(model.metrics_names, evaluation):
    print(f"{name}: {value}")

#predictions on validation set
predictions = model.predict([X_seq_val, X_static_val], verbose=0)  # shape: (num_samples, 3)
class_pred = np.argmax(predictions, axis=1)  # Convert probabilities to class predictions

#metrics on full validation set
report_full = classification_report(Y_val, class_pred, zero_division=1, digits=4, output_dict=True)

np.random.seed(42)

#identify indices of each class in Y_val
neither_indices_val = np.where(Y_val == 0)[0]
peak_indices_val = np.where(Y_val == 1)[0]
valley_indices_val = np.where(Y_val == 2)[0]

#determine the minimum count among the three classes
min_count = min(len(neither_indices_val), len(peak_indices_val), len(valley_indices_val))

#randomly select min_count indices from each class
neither_sample_indices = np.random.choice(neither_indices_val, size=min_count, replace=False)
peak_sample_indices = np.random.choice(peak_indices_val, size=min_count, replace=False)
valley_sample_indices = np.random.choice(valley_indices_val, size=min_count, replace=False)

#combine and shuffle
balanced_val_indices = np.concatenate([neither_sample_indices, peak_sample_indices, valley_sample_indices])
np.random.shuffle(balanced_val_indices)

#extract balanced subset
Y_val_bal = Y_val[balanced_val_indices]
predictions_bal = predictions[balanced_val_indices]
class_pred_bal = np.argmax(predictions_bal, axis=1)

#compute metrics on the balanced subset
report_bal = classification_report(Y_val_bal, class_pred_bal, zero_division=1, digits=4, output_dict=True)

# original Mapping:
# 0: Neither (0)
# 1: Peak (1)
# 2: Valley (-1)

class_mapping = {0: '0 (Neither)', 1: '1 (Peak)', 2: '-1 (Valley)'}

def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    Plots a confusion matrix as a heatmap

    - cm (array-like): confusion matrix
    - classes (list): list of class labels
    - title (str): plot title
    - cmap: heatmap colormap
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

cm_full = confusion_matrix(Y_val, class_pred)
print("\nConfusion Matrix (Classes: 0=Neither, 1=Peak, 2=Valley) on Full Validation Set:")
print(cm_full)
plot_confusion_matrix(cm_full,
                      classes=[class_mapping[i] for i in range(3)],
                      title='Confusion Matrix - Full Validation Set')

cm_bal = confusion_matrix(Y_val_bal, class_pred_bal)
print("\nConfusion Matrix (Classes: 0=Neither, 1=Peak, 2=Valley) on Balanced Validation Subset:")
print(cm_bal)
plot_confusion_matrix(cm_bal,
                      classes=[class_mapping[i] for i in range(3)],
                      title='Confusion Matrix - Balanced Validation Subset')

print("\nClassification Report on Full Validation Set:")
print(classification_report(Y_val, class_pred, zero_division=1, digits=4, target_names=[class_mapping[0], class_mapping[1], class_mapping[2]]))

print("\nClassification Report on Balanced Validation Subset:")
print(classification_report(Y_val_bal, class_pred_bal, zero_division=1, digits=4, target_names=[class_mapping[0], class_mapping[1], class_mapping[2]]))
