# import numpy as np
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast
# from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# # Blackbox function
# def bbox_fcn_sine3(input_parameters):
#     input_parameters = input_parameters.reshape(1,3)
#     M = 10
#     dt = 0.2
#     t = np.array([np.arange(M) * dt])
#     output = (input_parameters[:,[0]] * np.sin(input_parameters[:,[1]] @ t) + input_parameters[:,[2]])
#     output = output.flatten()
#     return output

# # Generate dataset
# def generate_data(num_samples, N, M, bbox_function):
#     inputs = np.random.uniform(-1, 1, (num_samples, N))
#     outputs = np.array([bbox_function(x) for x in inputs])
#     return inputs, outputs

# class BboxDataset(Dataset):
#     def __init__(self, inputs, outputs, tokenizer):
#         self.inputs = inputs
#         self.outputs = outputs
#         self.tokenizer = tokenizer
#         self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) ##

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, index):
#         input_vec = self.inputs[index]
#         output_vec = self.outputs[index]
#         input_str = ' '.join([f"{x:.5f}" for x in input_vec])
#         output_str = ' '.join([f"{x:.5f}" for x in output_vec])
#         tokenized = self.tokenizer(f"{output_str} <sep> {input_str}", return_tensors='pt', padding=True, truncation=True, max_length=512) ##
#         return tokenized

# def main():
#     N = 3
#     M = 10
#     bbox_function = bbox_fcn_sine3
#     num_train_samples = 10000
#     num_test_samples = 1000

#     # Generate train and test data
#     train_inputs, train_outputs = generate_data(num_train_samples, N, M, bbox_function)
#     test_inputs, test_outputs = generate_data(num_test_samples, N, M, bbox_function)

#     # Initialize GPT-2 model
#     config = GPT2Config(vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12)
#     model = GPT2LMHeadModel(config)
#     tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

#     # Create dataset
#     train_dataset = BboxDataset(train_inputs, train_outputs, tokenizer)
#     test_dataset = BboxDataset(test_inputs, test_outputs, tokenizer)

#     # Define training arguments and trainer
#     training_args = TrainingArguments(
#         output_dir="./bbox_transformer",
#         overwrite_output_dir=True,
#         num_train_epochs=10,
#         per_device_train_batch_size=32,
#         per_device_eval_batch_size=32,
#         logging_steps=100,
#         save_steps=1000,
#         save_total_limit=2,
#         # fp16=True, ##
#     )

#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#     )

#     # Train and evaluate the model
#     trainer.train()
#     eval_results = trainer.evaluate()
#     print(eval_results)

# if __name__ == "__main__":
#     main()



import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

# The given blackbox function
def bbox_fcn_sine3(input_parameters):
    input_parameters = input_parameters.reshape(1, 3)
    M = 10
    dt = 0.2
    t = np.array([np.arange(M) * dt])
    output = (input_parameters[:, [0]] * np.sin(input_parameters[:, [1]] @ t) + input_parameters[:, [2]])
    output = output.flatten()
    return output

# Function to generate input-output pairs
def generate_data(num_samples, N=3, low=-1, high=1):
    X = np.random.uniform(low, high, size=(num_samples, N))
    Y = np.array([bbox_fcn_sine3(x) for x in X])
    return X, Y

# Custom dataset class
class NumericArrayDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

# Function to pad and pack a batch
def pad_and_pack(batch):
    X, Y = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=-1)
    Y_padded = pad_sequence(Y, batch_first=True, padding_value=-1)
    return X_padded, Y_padded

# Transformer model
class NumericTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super(NumericTransformer, self).__init__()
        self.src_norm = nn.LayerNorm(d_model)
        self.tgt_norm = nn.LayerNorm(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, dim_feedforward)
        self.fc_in = nn.Linear(input_dim, d_model)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = self.fc_in(src)
        tgt = self.fc_in(tgt)
        src = self.src_norm(src)
        tgt = self.tgt_norm(tgt)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

def main():
    N = 3
    M = 10
    num_samples = 10000
    batch_size = 64
    epochs = 10

    # Generate data
    X_train, Y_train = generate_data(num_samples)
    X_test, Y_test = generate_data(1000)

    # Create datasets and data loaders
    train_dataset = NumericArrayDataset(X_train, Y_train)
    test_dataset = NumericArrayDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_and_pack)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_and_pack)

    # Initialize the model, loss function, and optimizer
    model = NumericTransformer(N, M)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_idx, (src, tgt) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            print(f"Train Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {loss.item():.6f}")

    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            output = model(src, tgt)
            loss = criterion(output, tgt)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.6f}")

if __name__ == "__main__":
    main()
