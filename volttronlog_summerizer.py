# %%
from datasets import load_dataset, Dataset
import pandas as pd

df11 = pd.read_parquet('volttronlog_mass.parquet')
df12= pd.read_parquet('540log_mass.parquet')
df21 = pd.read_parquet('volttronlog.parquet')
df22= pd.read_parquet('540log.parquet')
df=pd.concat([df11,df12,df21,df22],ignore_index=True)
df['response']=df['response'].apply(lambda array: array[0] )
raw=Dataset.from_pandas(df)
raw1=raw.train_test_split(test_size=0.2)
print(raw1)

# %%
from transformers import AutoTokenizer

checkpoint = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# %%
prefix = "summarize:"


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["context"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["response"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# %%
tokenized_billsum = raw1.map(preprocess_function, batched=True)

# %%
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# %%
import evaluate

rouge = evaluate.load("rouge")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# %%
from peft import LoraConfig, TaskType

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

# %%
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


# %%
text = "summarize:  2024-06-18 01:03:00,001 (platform_driveragent-4.0 878936 [223]) platform_driver.driver DEBUG: fake-campus/fake-building/fake-device next scrape scheduled: 2024-06-18 06:03:05+00:00"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=280)
print(tokenizer.batch_decode(outputs)[0])

# %%
from peft import get_peft_model
from peft import prepare_model_for_kbit_training
model.train() # model in training mode (dropout modules are activated)

# enable gradient check pointing
#model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)


model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %%
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="your-name/bigscience/mt0-large-lora",
    learning_rate=1e-10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
     weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=5,
    warmup_steps=5,
    fp16=True,
    optim="paged_adamw_8bit",
)

# %%
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    
)

trainer.train()

# %%
training_args = TrainingArguments(
    output_dir="your-name/bigscience/mt0-large-lora",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
     weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=50,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
)

# %%
text = "summerize discribe the events : 2024-06-18 01:03:00,001 (platform_driveragent-4.0 878936 [223]) platform_driver.driver DEBUG: fake-campus/fake-building/fake-device next scrape scheduled: 2024-06-18 06:03:05+00:00"

# %%
model.eval()
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
print(tokenizer.batch_decode(outputs)[0])



