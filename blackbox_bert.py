import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertConfig, BertModel, PreTrainedModel, Trainer, TrainingArguments

# Blackbox function
def bbox_fcn_sine3(input_parameters):
    input_parameters = input_parameters.reshape(1, 3)
    M = 10
    dt = 0.2
    t = np.array([np.arange(M) * dt])
    output = (input_parameters[:, [0]] * np.sin(input_parameters[:, [1]] @ t) + input_parameters[:, [2]])
    output = output.flatten()
    return output

# Dataset generation class
class BlackboxDataset(Dataset):
    def __init__(self, num_samples, bbox_function):
        self.num_samples = num_samples
        self.bbox_function = bbox_function
        self.data = []
        self.generate_data()

    def generate_data(self):
        for _ in range(self.num_samples):
            input_parameters = np.random.uniform(-1, 1, size=3)
            output = self.bbox_function(input_parameters)
            self.data.append((input_parameters, output))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

# Custom BERT model
class CustomBertModel(PreTrainedModel):
    def __init__(self, config, N, M):
        super().__init__(config)
        self.bert = BertModel(config)
        self.input_layer = nn.Linear(M, config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, N)
        self.init_weights()

    def forward(self, input_data):
        input_data = self.input_layer(input_data)
        outputs = self.bert(inputs_embeds=input_data)
        pooled_output = outputs.last_hidden_state[:, 0]
        output_data = self.output_layer(pooled_output)
        return output_data

# Data collator
def data_collator(batch): # 
    input_data, target_data = zip(*batch)
    input_data = torch.tensor(input_data, dtype=torch.float32)
    target_data = torch.tensor(target_data, dtype=torch.float32)
    return input_data, target_data

# Evaluation metric
def compute_mse(pred, labels):
    pred, labels = pred.float(), labels.float()
    return {'mse': nn.MSELoss()(pred, labels)}

# Main program
def main():
    N = 3
    M = 10
    num_samples = 1000
    train_ratio = 0.8
    config = BertConfig()

    # Generate dataset
    dataset = BlackboxDataset(num_samples, bbox_fcn_sine3)

    # Split dataset
    train_size = int(train_ratio * num_samples)
    test_size = num_samples - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create a custom BERT model
    model = CustomBertModel(config, N, M)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        logging_dir='./logs',
        no_cuda=True,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_mse,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

if __name__ == '__main__':
    main()


    