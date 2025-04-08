# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %%

# Load the SST-2 dataset
sst2_dataset = load_dataset("glue", "sst2")
train_dataset = sst2_dataset["train"]
validation_dataset = sst2_dataset["validation"]
test_dataset = sst2_dataset["test"]

# %%

model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%

# Define max sequence length
max_length = 128

# Custom dataset class compatible with HF Trainer
class QwenSentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Format examples: "Review: {text} Sentiment:"
        text = f"Review: {self.dataset[idx]['sentence']} Sentiment:"
        label = self.dataset[idx]['label']

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        # Extract input_ids and attention_mask and convert to appropriate format
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label
        }

# Create custom datasets
train_data = QwenSentimentDataset(train_dataset, tokenizer, max_length)
val_data = QwenSentimentDataset(validation_dataset, tokenizer, max_length)
# Create data loaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# %%


# Define the Qwen with classification head model
class QwenForSentimentClassification(nn.Module):
    def __init__(self, model_id, num_classes=2, unfreeze_layers=2):
        super(QwenForSentimentClassification, self).__init__()

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load pre-trained Qwen model
        self.qwen = AutoModel.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Prepare model for k-bit training
        self.qwen = prepare_model_for_kbit_training(self.qwen)

        # Get Qwen configuration
        config = self.qwen.config

        # Add classification head
        hidden_size = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

        # Set up LoRA configuration
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=16,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )

        # Apply LoRA to the model
        self.qwen = get_peft_model(self.qwen, peft_config)

        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        # Get Qwen outputs
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)

        # Use the last hidden state of the last token for classification
        last_hidden_state = outputs.last_hidden_state

        # Get the last token representation for each sequence
        # by using the attention mask to identify the last token
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_hidden = torch.stack([
            last_hidden_state[i, last_idx, :]
            for i, last_idx in enumerate(last_token_indices)
        ])

        # Pass through the classification head
        logits = self.classifier(last_token_hidden)

        # Calculate loss if labels are provided (during training)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

        return {"logits": logits}

# %%

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = QwenForSentimentClassification(model_id=model_id, num_classes=2, unfreeze_layers=2).to(device)

# %%

# Initialize training arguments
training_args = TrainingArguments(
    output_dir="./qwen-sentiment-classifier",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-4,              # Slightly higher for LoRA
    eval_strategy="steps",
    eval_steps=100,                  # Evaluate every 100 steps
    save_strategy="steps",           # Save checkpoints by steps
    save_steps=100,                  # Save every 100 steps
    save_total_limit=3,              # Keep only the 3 most recent checkpoints
    logging_dir="./logs",
    logging_steps=50,                # Log every 50 steps
    num_train_epochs=3,
    load_best_model_at_end=True,     # Load the best model at the end of training
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=False,                     # Disable FP16 since using 4-bit quantization
    bf16=True,                      # Enable BF16 which is compatible with 4-bit training
    gradient_accumulation_steps=4,  # Accumulate gradients for effective batch size = 16*4=64
    warmup_steps=100,               # Warm up learning rate for first 100 steps
    weight_decay=0.01,              # Weight decay for regularization
)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions)
    }

# Initialize and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

# Check if a checkpoint exists before attempting to resume
checkpoint_dir = "./qwen-sentiment-classifier"
resume_from_checkpoint = os.path.isdir(checkpoint_dir) and any("checkpoint" in f for f in os.listdir(checkpoint_dir))

# Train the model (with resumption capability)
trainer.train(resume_from_checkpoint=resume_from_checkpoint)

# Save the final model
trainer.save_model("./qwen-sentiment-final")
print(f"Best model saved at: {trainer.state.best_model_checkpoint}")

"""
The validation accuracy was slightly over 97% after 3 epochs.
"""

# %%

# Define inference function for new text
def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()

    # Format the input
    formatted_text = f"Review: {text} Sentiment:"

    # Tokenize
    encoding = tokenizer(formatted_text,
                         return_tensors='pt',
                         max_length=max_length,
                         padding='max_length',
                         truncation=True)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)["logits"]
        _, preds = torch.max(logits, dim=1)

    # Map prediction to sentiment
    sentiment = "positive" if preds.item() == 1 else "negative"

    return sentiment

# Example usage
sample_texts = [
    "As personal and egoless as you could ever hope to expect from an $120 million self-portrait that doubles as a fable about the fall of Ancient Rome, Francis Ford Coppola’s “Megalopolis” is the story of an ingenious eccentric who dares to stake his fortune on a more optimistic vision for the future — not because he thinks he can single-handedly bring that vision to bear, but rather because history has taught him that questioning a civilization’s present condition is the only reliable hope for preventing its ruin. Needless to say, the movie isn’t arriving a minute too soon.",
    "YALL ARE WRONG You see you gotta go into the movie the same way as Francis Ford Coppola, blazed out of your fucking mind. You gotta meet him on the same plane, the same level of thinking. Hitting the cart 6 times before entering the showing like I did.",
    "the 138-minute cinematic equivalent of someone showing you a youtube video they promise is really good",
    "Dropping the kids off at Oppenheimer so the adults could watch Barbie",
    "Hi Barbie.",
    "As a work of and about plasticity, Barbie succeeds with vibrant, glowing colors. This is a triumph of manufactured design, with its expansive dollhouses and fashion accessories complimenting the mannered, almost manicured narrative journey of toys and humans.",
    "I’m sorry to talk about a man when this is very much a movie about women but every second Gosling is onscreen is so funny. Even when he’s just standing there not talking it’s funny.",
    "A ridiculous achievement in filmmaking. An absurdly immersive and heart-pounding experience. Cillian Murphy is a fucking stud and RDJ will be a front-runner for Best Supporting Actor. Ludwig Göransson put his entire nutsack into that score, coupled with a sound design that made me feel like I took a bomb to the chest."
]
for sample_text in sample_texts:
    prediction = predict_sentiment(sample_text, model, tokenizer, device)
    print(f"Sample text: '{sample_text}'")
    print(f"Predicted sentiment: {prediction}")

"""
Kernel output:

Sample text: 'As personal and egoless as you could ever hope to expect from an $120 million self-portrait that doubles as a fable about the fall of Ancient Rome, Francis Ford Coppola’s “Megalopolis” is the story of an ingenious eccentric who dares to stake his fortune on a more optimistic vision for the future — not because he thinks he can single-handedly bring that vision to bear, but rather because history has taught him that questioning a civilization’s present condition is the only reliable hope for preventing its ruin. Needless to say, the movie isn’t arriving a minute too soon.'
Predicted sentiment: positive
Sample text: 'YALL ARE WRONG You see you gotta go into the movie the same way as Francis Ford Coppola, blazed out of your fucking mind. You gotta meet him on the same plane, the same level of thinking. Hitting the cart 6 times before entering the showing like I did.'
Predicted sentiment: negative
Sample text: 'the 138-minute cinematic equivalent of someone showing you a youtube video they promise is really good'
Predicted sentiment: negative
Sample text: 'Dropping the kids off at Oppenheimer so the adults could watch Barbie'
Predicted sentiment: negative
Sample text: 'Hi Barbie.'
Predicted sentiment: positive
Sample text: 'As a work of and about plasticity, Barbie succeeds with vibrant, glowing colors. This is a triumph of manufactured design, with its expansive dollhouses and fashion accessories complimenting the mannered, almost manicured narrative journey of toys and humans.'
Predicted sentiment: positive
Sample text: 'I’m sorry to talk about a man when this is very much a movie about women but every second Gosling is onscreen is so funny. Even when he’s just standing there not talking it’s funny.'
Predicted sentiment: positive
Sample text: 'A ridiculous achievement in filmmaking. An absurdly immersive and heart-pounding experience. Cillian Murphy is a fucking stud and RDJ will be a front-runner for Best Supporting Actor. Ludwig Göransson put his entire nutsack into that score, coupled with a sound design that made me feel like I took a bomb to the chest.'
Predicted sentiment: positive

"""
