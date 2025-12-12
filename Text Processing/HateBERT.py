!pip install transformers datasets pyarrow huggingface_hub cleantext contractions
!pip install datasets

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import contractions
import cleantext
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification
import tensorflow as tf

!pip install huggingface_hub
!pip install pyarrow

import zipfile
import os

zip_file_path = '/content/archive (12).zip'
extract_dir = '/content/extracted_data'

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"File '{zip_file_path}' unzipped to '{extract_dir}'")
print(f"Contents of '{extract_dir}':")
for item in os.listdir(extract_dir):
    print(item)

DATA_PATH = "/content/extracted_data/labeled_data.csv"

pandas_df = pd.read_csv(DATA_PATH)
pandas_df.head()

import seaborn as sns
sns.countplot(x='class', data=pandas_df)

import re
import contractions
import cleantext

# Regex pattern to remove emojis
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def clean_tweet(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Remove mentions
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 4. Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # 5. Expand contractions
    text = contractions.fix(text)

    # 6. Remove hashtags symbol (#)
    text = text.replace("#", "")

    # 7. Remove emojis using regex
    text = emoji_pattern.sub(r'', text)

    # 8. Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # 9. Normalize elongated words
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    #10. Normalize numbers
    text = re.sub(r"\d+", " ", text)

    #11. Remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    #12. Fix spacing issues
    text = re.sub(r"\s+", " ", text).strip()

    return text

pandas_df["tweet_cleaned"] = pandas_df["tweet"].apply(clean_tweet)

train_df, temp_df = train_test_split(
    pandas_df, test_size=0.3, stratify=pandas_df["class"], random_state=42
)

valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["class"], random_state=42
)

train_data = Dataset.from_pandas(train_df)
valid_data = Dataset.from_pandas(valid_df)
test_data = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    "train": train_data,
    "valid": valid_data,
    "test": test_data
})

model_name = "GroNLP/hateBERT"  # best for toxic/hate speech tasks

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["tweet_cleaned"], truncation=True)

dataset_tokenized = dataset.map(tokenize, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

train_tf = dataset_tokenized["train"].to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["class"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator
)

valid_tf = dataset_tokenized["valid"].to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["class"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator
)

test_tf = dataset_tokenized["test"].to_tf_dataset(
    columns=tokenizer.model_input_names,
    label_cols=["class"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator
)

model = TFAutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=3, from_pt=True
)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(pandas_df["class"]),
    y=pandas_df["class"]
)

cw = {i: class_weights[i] for i in range(3)}

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

history = model.fit(
    train_tf,
    validation_data=valid_tf,
    epochs=5,
    class_weight=cw
)

loss, acc = model.evaluate(test_tf)
print("Final Test Accuracy:", acc)
