
import torch
import transformers
import json
from sklearn.model_selection import train_test_split
from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments
from transformers import GPT2TokenizerFast

from torch.utils.data import DataLoader
from transformers import AdamW


def readsmiles(datafile):
    fp=open(datafile,"r")
    samples=[]
    labels=[]
    count=0
    for line in fp:
        if len(line)<5:
            continue
        # print(line)
        term=line.split()
    #     # print("term", term)
        samples.append(term)
    #     #print(len(line.split("$")))
    #     term=line.split("$")[1]
        
    #     if term.strip()=="0":
    #         label=0
    #     else:
    #         label=1
        
    #     sample=line.split("$")[0]
    #     # sample=sample+"$"
    #     samples.append(sample)
    #     # print(sample)
    #     count=count+1    
    #     labels.append(label)
    # return samples, labels

        
        # if term.strip()=="0":
        #     label=0
        # else:
        #     label=1
        
        # sample=line.split("$")[0]
        # # sample=sample+"$"
        # samples.append(sample)
        # # print(sample)
        # count=count+1    
        # labels.append(label)
        # print(samples)
    return samples

class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from transformers.trainer import DataCollatorWithPadding


# from transformers import DataCollatorWithPadding



# for fold in range(1): #smiles_dict.keys():

   
#     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
#     tokenizer.pad_token = tokenizer.eos_token
#     train_dataset, val_dataset, test_dataset = generate_dataset(tokenizer, './smiles_total_4.txt', './test_smiles4.txt')
#     model=GPT2ForSequenceClassification.from_pretrained("gpt2")
#     model.config.pad_token_id = model.config.eos_token_id
#     model.to(device)
#     #print(train_dataset)
#     training_args = TrainingArguments(
#     output_dir='./results_disc_new_single_4',          # output directory
#     overwrite_output_dir = True ,
#     num_train_epochs=5,              # total number of training epochs
#     per_device_train_batch_size=1,  # batch size per device during training
#     per_device_eval_batch_size=1,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs_disc_new_single_4',            # directory for storing logs
#     logging_steps=10,
#     save_total_limit=5
#     )


#     trainer = Trainer(
#     model=model,                         # the instantiated model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=val_dataset ,  
#     data_collator=data_collator          # evaluation dataset
#     )

    # trainer.train()
    # torch.save(model,"gpt2_disc_4.pt")

# #model.train()
    

# """train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


# optim = AdamW(model.parameters(), lr=5e-5)

# for epoch in range(3):
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs[0]
#         loss.backward()
#         optim.step()

# model.eval()"""
from transformers import pipeline
from tqdm.auto import tqdm
fold = 0 
test_samples = readsmiles('./results_smiles_gpt2_10000_1.txt')
print(test_samples)
#model=torch.load("fold"+str(fold)+"gpt2_single.pt")
#device=torch.device("cpu")
#from transformers.pipelines.pt_utils import KeyDataset
# model.cuda()
model=torch.load("./gpt2_disc_4.pt")
from transformers import pipeline
from tqdm.auto import tqdm
device=torch.device("cuda")
#from transformers.pipelines.pt_utils import KeyDataset
#model.cuda()
# tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2', truncation=True)
# tokenizer.pad_token = tokenizer.eos_token

# generator = pipeline(task="text-generation",  model=model, tokenizer=tokenizer)
# tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
# generator("0")
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
# tokenizer.pad_token = tokenizer.eos_token
classifier = pipeline(task="text-classification", model=model.to('cpu'), tokenizer=tokenizer)
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
predictions=[]
import csv

# Define the file path where the CSV file will be written
csv_file = './proxy_predictions_2.csv'

# Define the column names for the CSV file
columns = ['smiles', 'label']
with open(csv_file, mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for i in range(len(test_samples)):
        predictions = classifier(str(test_samples[i]), **tokenizer_kwargs)[0]
        # print(predictions)
        row = [str(test_samples[i]), str(predictions['label'])]
        writer.writerow(row)
    # print(predictions[0])


from evaluate import evaluator
import datasets
from datasets import Dataset
    
# train_samples, train_labels = readsmiles('./train_smiles_currentfold.txt')
# test_samples, test_labels = readsmiles('./test_smiles_currentfold.txt')
# train_samples, val_samples, train_labels, val_labels = train_test_split(train_samples, train_labels, test_size=0.1)
# train_encodings = tokenizer(train_samples, truncation=True, padding = True)
# val_encodings = tokenizer(val_samples, truncation = True, padding = True)
# test_encodings = tokenizer(test_samples, truncation = True, padding = True)
# import pyarrow as pa
# import pandas as pd 
# df = pd.read_csv('./test_smiles_currentfold.csv')
# # ds = Dataset(df)
# def convert_to_table(smiles_dataset):
#     # Extract the data from SmilesDataset object
#     data = smiles_dataset.data
#     # Convert the data into a pyarrow Table object
#     table = pa.Table.from_arrays(data, names=smiles_dataset.columns)
#     return table

# smiles_dataset = SmilesDataset(test_encodings, test_labels)
# table = convert_to_table(df)
# ds = Dataset(table)
# from datasets import load_dataset
# dataset = load_dataset('csv', data_files = {'train': ['./smiles_total_withsplit.csv'], 'test': './test_smiles_withsplit.csv'})
# print(dataset['train'][0])
# results = task_evaluator.compute(

#     model_or_pipeline=model,
#     tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True),

#     data=dataset['test'],

#     metric="f1",

#     label_mapping={"LABEL_0": 0.0, "LABEL_1": 1.0},

# )
# print(results)





