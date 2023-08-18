
import torch
torch.cuda.empty_cache()
import transformers

import json
from sklearn.model_selection import train_test_split
from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments
from transformers import GPT2TokenizerFast

from torch.utils.data import DataLoader
from transformers import AdamW


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def readsmiles(datafile):
    fp=open(datafile,"r")
    samples=[]
    labels=[]
    count=0
    for line in fp:
        if len(line)<5:
            continue
        # print(line)
        # term=line.split("$")[1]
        
        # if term.strip()=="-1":
        #     label=0
        # else:
        #     label=1
        
        # sample=line.split("$")[0]
        sample=line.split("\n")[0]
        # sample=sample+"$"
        sample=sample
        samples.append(sample)
        count=count+1    
        labels.append(count)
    return samples, labels
		
train_samples, train_labels = readsmiles('')
test_samples, test_labels = readsmiles('')
# print("samples", train_samples, "labels", train_labels)

# from sklearn.model_selection import train_test_split
train_samples, val_samples, train_labels, val_labels = train_test_split(train_samples, train_labels, test_size=0.1)

from transformers import GPT2TokenizerFast
from transformers import BioGptTokenizer, BioGptForCausalLM
# from transformers import AutoTokenizer, T5ForConditionalGeneration
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2', truncation=True)
# from transformers import AutoTokenizer, AutoModelForMaskedLM
# # tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
# tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
# # tokenizer = AutoTokenizer.from_pretrained("mrm8488/chEMBL26_smiles_v2")
# tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")


# # tokenizer = AutoTokenizer.from_pretrained("laituan245/molt5-small", model_max_length=512)


tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# train_encodings = tokenizer(train_samples)
# val_encodings = tokenizer(val_samples)
# test_encodings = tokenizer(test_samples)
train_encodings = tokenizer(train_samples, truncation=True, padding = True)
val_encodings = tokenizer(val_samples, truncation = True, padding = True)
test_encodings = tokenizer(test_samples, truncation = True, padding = True)


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

train_dataset = SmilesDataset(train_encodings, train_labels)
val_dataset = SmilesDataset(val_encodings, val_labels)
test_dataset = SmilesDataset(test_encodings, test_labels)


# # print(train_dataset)
# from transformers import  Trainer, TrainingArguments
# #model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
model=AutoModelForCausalLM.from_pretrained("distilgpt2")
# # model = AutoModelForCausalLM.from_pretrained("mrm8488/chEMBL26_smiles_v2")
# # model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
# model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
# # model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-small')
# # model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-4.7M")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# torch.cuda.set_device(7)
model.to(device)

training_args = TrainingArguments(
    output_dir='./results_gen_single_gpt2_all', 
    overwrite_output_dir = True ,      # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs_gen_single_gpt2_all',            # directory for storing logs
    logging_steps=10,
    resume_from_checkpoint='./results_gen_single_gpt2_all/checkpoint-681500',
    save_total_limit=5
    
)


trainer = Trainer(
    model=model,                         # the instantiated model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,             # evaluation dataset
    data_collator=data_collator
)

# trainer.train('')
# fold = 1
# torch.save(model, str(fold)+"gpt2_all.pt")
from torch.utils.data import DataLoader
from transformers import AdamW

"""train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()"""



model=torch.load("")
from transformers import pipeline
from tqdm.auto import tqdm
# device=torch.device("cuda")
# # model.to(device)

# # from transformers.pipelines.pt_utils import KeyDataset
# #model.cuda()
# # import csv 
# model_name = ""
# model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline(task="text-generation",  model=model.to('cpu'), tokenizer=tokenizer, temperature=0.8)
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':2048}
num_generations = 26

generated_text = [generator(" ")[0]['generated_text'] for i in range(10000)]
with open('results_smiles_gpt2_10000_1.txt', 'w') as file:
    for text_and_class in generated_text:
        file.write(text_and_class + "\n")
k = 5  # set the value of k
# with torch.no_grad():
#     input_ids = tokenizer("O=C(NO)c1cccc(OCc2ccc(-c3ccccc3)cc2)", return_tensors='pt').input_ids.repeat(num_generations, 1)
#     logits = model(input_ids.to(model.device)).logits
#     top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

# # Print the top-k generated text for each prompt
# for i, prompt in enumerate(generated_text):
#     print(f"Prompt {i}: {prompt}")
#     for j in range(k):
#         generated_sequence = tokenizer.decode(top_k_indices[i][j], skip_special_tokens=True)
#         print(f"Top-{j+1} sequence: {generated_sequence}")
#     print("")
# with open('generated_text.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Generated Text"])
#     for text in generated_text:
#         writer.writerow([text])

#evaluating the classifier 



# generator = pipeline(task="text-generation",  model=model, tokenizer=tokenizer)
# tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
# generator("0 $ ")
# from transformers import pipeline
# from tqdm.auto import tqdm
# model=torch.load("")
# test_samples, test_labels = readsmiles('')
# # for i in range(50):
# #     print(test_samples[i])
# classifier = pipeline(task="text-classification",  model=model, tokenizer=tokenizer)
# tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
# predictions=[]
# for i in range(50):
#     predictions.append(classifier(test_samples[i],**tokenizer_kwargs))
# print(predictions)


