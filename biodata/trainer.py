from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from transformers import EarlyStoppingCallback

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)


labels = ["narcissistic","toxic","cringe","healthy"]
num_labels = len(labels)


def tokenize_and_format(examples):

    tokenized = tokenizer(examples["bio"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = [
        [float(examples[l][i]) for l in labels] 
        for i in range(len(examples["bio"]))
    ]
    return tokenized


df = pd.read_csv("/Users/pashantraj/Desktop/Repos/csea/CSEA-Inductions/dating_bios.csv")
dataset = Dataset.from_pandas(df).map(tokenize_and_format, batched=True)
train_test = dataset.train_test_split(test_size=0.2)


model = DistilBertForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,
    problem_type="multi_label_classification"
)


training_args = TrainingArguments(
    output_dir="./flag_detector_results",
    num_train_epochs=10,              
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2, 
    load_best_model_at_end=True,      
    metric_for_best_model="eval_loss",
    greater_is_better=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
)



trainer.train()
model.save_pretrained("final_flag_model")
tokenizer.save_pretrained("final_flag_model")


