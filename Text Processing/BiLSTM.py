import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
except:
    print("NLTK downloads completed or already available")

class HateSpeechBiLSTM:
    def __init__(self, max_features=20000, max_len=100, embedding_dim=100, lstm_units=128):
        self.max_features = max_features
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False

    def clean_text(self, text):
        """
        Clean and preprocess text data
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove user mentions and hashtags (but keep the text content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)

        # Remove punctuation and numbers but keep basic punctuation for context
        text = re.sub(r'[^\w\s!?]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize and remove stopwords
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words
                if word not in self.stop_words and len(word) > 2]

        return ' '.join(words)

    def load_and_prepare_data(self, file_path, text_column='text', label_column='label'):
        """
        Load dataset from CSV file and prepare it for training
        """
        print(f"Loading dataset from {file_path}...")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Check if required columns exist
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {list(df.columns)}")
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset. Available columns: {list(df.columns)}")

        print(f"Dataset loaded successfully! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Display dataset info
        print("\nDataset Info:")
        print(f"Total samples: {len(df)}")
        print(f"Label distribution:")
        print(df[label_column].value_counts())

        # Extract texts and labels
        texts = df[text_column].fillna('').astype(str).tolist()
        labels = df[label_column].astype(str).tolist()

        # Clean texts
        print("\nCleaning texts...")
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Remove empty texts after cleaning
        non_empty_indices = [i for i, text in enumerate(cleaned_texts) if text.strip()]
        cleaned_texts = [cleaned_texts[i] for i in non_empty_indices]
        labels = [labels[i] for i in non_empty_indices]

        print(f"After cleaning: {len(cleaned_texts)} samples")

        # Encode labels
        print("\nEncoding labels...")
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)

        # Show label mapping
        print("Label mapping:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name} -> {i}")

        # Tokenize texts
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(cleaned_texts)

        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)

        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_len)

        print(f"Final data shape: {X.shape}")
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")

        return X, encoded_labels, cleaned_texts

    def build_model(self, num_classes):
        """
        Build BiLSTM model for text classification
        """
        self.model = Sequential([
            # Embedding layer
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
            SpatialDropout1D(0.2),

            # First BiLSTM layer
            Bidirectional(LSTM(self.lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            BatchNormalization(),

            # Second BiLSTM layer
            Bidirectional(LSTM(self.lstm_units // 2, dropout=0.2, recurrent_dropout=0.2)),
            BatchNormalization(),

            # Dense layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(32, activation='relu'),
            Dropout(0.2),

            # Output layer
            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("BiLSTM Model architecture:")
        self.model.summary()

        return self.model

    def build_simple_bilstm(self, num_classes):
        """
        Build a simpler BiLSTM model (faster training)
        """
        self.model = Sequential([
            Embedding(self.max_features, self.embedding_dim, input_length=self.max_len),
            SpatialDropout1D(0.3),

            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            BatchNormalization(),

            Dense(64, activation='relu'),
            Dropout(0.3),

            Dense(32, activation='relu'),
            Dropout(0.2),

            Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("Simple BiLSTM Model architecture:")
        self.model.summary()

        return self.model

    def train(self, file_path, text_column='text', label_column='label',
              epochs=25, batch_size=32, validation_split=0.2, simple_model=False):
        """
        Train the BiLSTM model with your dataset
        """
        # Load and prepare data
        X, y, texts = self.load_and_prepare_data(file_path, text_column, label_column)

        # Build model
        num_classes = len(self.label_encoder.classes_)
        if simple_model:
            self.build_simple_bilstm(num_classes)
        else:
            self.build_model(num_classes)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )

        print(f"\nTraining samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Callbacks
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
            ModelCheckpoint('best_bilstm_model.h5', save_best_only=True, monitor='val_loss')
        ]

        # Train model
        print("\nStarting BiLSTM training...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Plot training history
        self.plot_training_history(history)

        return history

    def plot_training_history(self, history):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def predict(self, text):
        """
        Predict class for a single text
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        # Clean text
        cleaned_text = self.clean_text(text)

        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])

        # Pad sequence
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)

        # Predict
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction, axis=1)[0]

        # Decode class
        class_name = self.label_encoder.inverse_transform([predicted_class_idx])[0]

        # Get probabilities for all classes
        probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(self.label_encoder.classes_, prediction[0])
        }

        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'class': class_name,
            'confidence': float(confidence),
            'probabilities': probabilities
        }

    def predict_batch(self, texts):
        """
        Predict classes for multiple texts
        """
        # Clean all texts
        cleaned_texts = [self.clean_text(text) for text in texts]

        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)

        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len)

        # Predict
        predictions = self.model.predict(padded_sequences, verbose=0)
        predicted_classes_idx = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        # Decode classes
        predicted_classes = self.label_encoder.inverse_transform(predicted_classes_idx)

        results = []
        for i, text in enumerate(texts):
            probabilities = {
                class_name: float(prob)
                for class_name, prob in zip(self.label_encoder.classes_, predictions[i])
            }

            results.append({
                'text': text,
                'cleaned_text': cleaned_texts[i],
                'class': predicted_classes[i],
                'confidence': float(confidences[i]),
                'probabilities': probabilities
            })

        return results

    def evaluate_model(self, file_path, text_column='text', label_column='label'):
        """
        Evaluate the model on a test dataset
        """
        print("Evaluating BiLSTM model on test data...")

        # Load and prepare data
        X, y, texts = self.load_and_prepare_data(file_path, text_column, label_column)

        # Predict
        y_pred_probs = self.model.predict(X, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=self.label_encoder.classes_))

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('BiLSTM Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Calculate additional metrics
        accuracy = np.mean(y_pred == y)
        print(f"Overall Accuracy: {accuracy:.4f}")

        return classification_report(y, y_pred, target_names=self.label_encoder.classes_, output_dict=True)

    def save_model(self, model_path='hate_speech_bilstm_model'):
        """
        Save the trained model and preprocessing objects
        """
        if not self.is_trained:
            raise ValueError("No trained model to save.")

        # Save Keras model
        self.model.save(f'{model_path}.h5')

        # Save tokenizer and label encoder
        joblib.dump(self.tokenizer, f'{model_path}_tokenizer.pkl')
        joblib.dump(self.label_encoder, f'{model_path}_label_encoder.pkl')

        # Save model configuration
        config = {
            'max_features': self.max_features,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units
        }
        joblib.dump(config, f'{model_path}_config.pkl')

        print(f"BiLSTM model saved successfully to {model_path}.h5")
        print(f"Preprocessing objects saved to {model_path}_*.pkl")

    def load_model(self, model_path='hate_speech_bilstm_model'):
        """
        Load a trained model and preprocessing objects
        """
        # Load model configuration
        config = joblib.load(f'{model_path}_config.pkl')
        self.max_features = config['max_features']
        self.max_len = config['max_len']
        self.embedding_dim = config['embedding_dim']
        self.lstm_units = config.get('lstm_units', 128)

        # Load tokenizer and label encoder
        self.tokenizer = joblib.load(f'{model_path}_tokenizer.pkl')
        self.label_encoder = joblib.load(f'{model_path}_label_encoder.pkl')

        # Load Keras model
        self.model = load_model(f'{model_path}.h5')
        self.is_trained = True

        print(f"BiLSTM model loaded successfully from {model_path}.h5")

# Comparison class to compare CNN and BiLSTM
class ModelComparator:
    def __init__(self):
        self.models = {}
        self.results = {}

    def add_model(self, name, model):
        self.models[name] = model

    def compare_predictions(self, texts):
        comparisons = []
        for text in texts:
            result = {'text': text}
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    prediction = model.predict(text)
                    result[name] = {
                        'class': prediction['class'],
                        'confidence': prediction['confidence']
                    }
            comparisons.append(result)
        return comparisons

# Example usage with your dataset
def main():
    # Initialize the BiLSTM hate speech detector
    detector = HateSpeechBiLSTM(
        max_features=20000,  # Adjust based on your dataset size
        max_len=100,         # Adjust based on your text length
        embedding_dim=100,
        lstm_units=128
    )

    # Train the model with your dataset
    print("=== BiLSTM TRAINING MODE ===")
    history = detector.train(
        file_path='your_dataset.csv',  # Replace with your dataset path
        text_column='text',           # Replace with your text column name
        label_column='label',         # Replace with your label column name
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        simple_model=False  # Set to True for faster training
    )

    # Save the trained model
    detector.save_model('my_hate_speech_bilstm_model')

    # Test with some examples
    print("\n=== BiLSTM TESTING MODE ===")
    test_texts = [
        "This is a wonderful day!",
        "You are all terrible people",
        "I don't like this very much",
        "They should all be eliminated",
        "Great work everyone!",
        "I hate you so much",
        "This is absolutely disgusting behavior",
        "What a beautiful morning"
    ]

    print("BiLSTM Predictions on test texts:")
    for text in test_texts:
        result = detector.predict(text)
        print(f"\nText: '{text}'")
        print(f"Cleaned: '{result['cleaned_text']}'")
        print(f"Prediction: {result['class']} (Confidence: {result['confidence']:.3f})")
        print(f"All probabilities: {result['probabilities']}")

# If you want to use the model without training each time
def load_and_use_pretrained():
    """
    Load a pre-trained BiLSTM model and use it for predictions
    """
    detector = HateSpeechBiLSTM()

    try:
        detector.load_model('my_hate_speech_bilstm_model')
        print("Pre-trained BiLSTM model loaded successfully!")

        # Your text to classify
        your_text = "This is a test message to classify"
        result = detector.predict(your_text)

        print(f"Text: {result['text']}")
        print(f"Classification: {result['class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"All probabilities: {result['probabilities']}")

    except FileNotFoundError:
        print("No pre-trained BiLSTM model found. Please train the model first.")

# Batch prediction example
def batch_prediction_example():
    """
    Example of batch prediction with BiLSTM
    """
    detector = HateSpeechBiLSTM()
    detector.load_model('my_hate_speech_bilstm_model')

    # Multiple texts to classify
    texts_to_classify = [
        "I love this community!",
        "You people are disgusting",
        "This is okay I guess",
        "I wish harm upon all of them",
        "Nice weather today"
    ]

    print("Batch Prediction Results:")
    results = detector.predict_batch(texts_to_classify)

    for result in results:
        print(f"\nText: {result['text']}")
        print(f"→ Class: {result['class']} (Confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    # Run the main training and testing
    main()

    # Or uncomment below to just load and use a pre-trained model
    # load_and_use_pretrained()

    # Or uncomment for batch prediction example
    # batch_prediction_example()
