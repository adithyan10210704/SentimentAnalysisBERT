from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Drop text column and set format
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="no"
)

# Prepare Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(5000)),
    eval_dataset=tokenized_dataset["test"].select(range(1000)),
)

# Train and evaluate
trainer.train()
trainer.evaluate()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

text = "This movie was absolutely fantastic!"

# Tokenize and move to device
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

# Prediction
with torch.no_grad():
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1)

# Print sentiment
print("Sentiment:", "Positive" if pred.item() == 1 else "Negative")

model.save_pretrained("sentiment_model")
tokenizer.save_pretrained("sentiment_model")
