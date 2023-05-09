## please install the libraries in the imports using pip commands
## use anaconda prompt for best compatability 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
import numpy as np
import math
import random
import itertools
import pandas as pd

class SRLDataset(Dataset):
    def __init__(self, sentences, predicates, labels, tokenizer, max_length):
        self.sentences = sentences
        self.predicates = predicates
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        predicate = self.predicates[index]
        label = self.labels[index]

        # Tokenize sentence without special tokens to handle alignment
        tokenized_sentence = self.tokenizer.tokenize(sentence)

        # Initialize a list of labels with -100 (ignored by loss function) and the same length as the tokenized sentence
        aligned_labels = [-100] * len(tokenized_sentence)

        # Iterate through the original sentence words, labels, and their indices
        words = sentence.split()
        for word, lbl, idx in zip(words, label, range(len(words))):
            # Tokenize the current word
            subwords = self.tokenizer.tokenize(word)

            # Assign the label to the first subword of the current word
            subword_idx = tokenized_sentence.index(subwords[0], idx)
            aligned_labels[subword_idx] = lbl

        # Tokenize sentence and add [CLS] and [SEP] tokens
        tokenized_sentence = self.tokenizer.encode(sentence, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(tokenized_sentence, dtype=torch.long)

        # Add [CLS] and [SEP] tokens to the aligned_labels and pad or truncate to match max_length
        aligned_labels = [-100] + aligned_labels[:self.max_length - 2] + [-100]
        aligned_labels = aligned_labels + [-100] * (self.max_length - len(aligned_labels))

        # Convert the aligned_labels list to a torch tensor
        aligned_labels = torch.tensor(aligned_labels, dtype=torch.long)

        # Find index of predicate in tokenized sentence
        predicate_idx = tokenized_sentence.index(self.tokenizer.encode(predicate)[1])

        return input_ids, predicate_idx, aligned_labels

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def convert_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    sentences = []
    tags = []
    predicates = []
    sentence = []
    tag = []
    curr_pred = None
    exists_pred_in_sent = False
    exists_arg_in_sent = False

    for line in lines:
        if len(line) != 0:
            sentence.append(line[0])

            if len(line) >= 6:
                if line[5] == "PRED":
                    curr_pred = line[0]
                    exists_pred_in_sent = True

                if line[5] == "ARG1":
                    exists_arg_in_sent = True
                    tag.append(1)
                else:
                    tag.append(0)
            else:
                tag.append(0)
        else:
            if exists_arg_in_sent and exists_pred_in_sent:
                sentences.append(" ".join(sentence))
                tags.append(tag)
                predicates.append(curr_pred)

                exists_pred_in_sent = False
                exists_arg_in_sent = False

            sentence = []
            tag = []
            curr_pred = None

    if len(sentence) > 0 and exists_arg_in_sent and exists_pred_in_sent:
        sentences.append(" ".join(sentence))
        tags.append(tag)
        predicates.append(curr_pred)

    return sentences, tags, predicates



sentences_1, labels_1, predicates_1 = convert_file('partitive_group_nombank.clean.train')

sentences_2, labels_2, predicates_2 = convert_file('partitive_group_nombank.clean.test')

sentences_3, labels_3, predicates_3 = convert_file('partitive_group_nombank.clean.dev')

print(len(sentences_1))
print(len(labels_1))
print(len(predicates_1))



print(len(sentences_2))
print(len(labels_2))
print(len(predicates_2))

print(len(sentences_3))
print(len(labels_3))
print(len(predicates_3))

sentences_3[0]

labels_3[0]

predicates_1[0:5]

sentences_all = sentences_1.copy()
labels_all = labels_1.copy()
predicates_all = predicates_1.copy()

sentences_all.extend(sentences_2)
labels_all.extend(labels_2)
predicates_all.extend(predicates_2)

sentences_all.extend(sentences_3)
labels_all.extend(labels_3)
predicates_all.extend(predicates_3)

print(len(sentences_all))
print(len(labels_all))
print(len(predicates_all))

# Combine sentences, labels, and predicates into a list of tuples
combined_data = list(zip(sentences_all, labels_all, predicates_all))

# Shuffle the combined data using a random seed for reproducibility
random_seed = 42
random.seed(random_seed)
random.shuffle(combined_data)

# Split the shuffled data into training and validation sets
split_ratio = 0.8  
split_index = int(len(combined_data) * split_ratio)

train_data = combined_data[:split_index]
val_data = combined_data[split_index:]

# Separate sentences, labels, and predicates for the train and validation sets
train_sentences, train_labels, train_predicates = zip(*train_data)
val_sentences, val_labels, val_predicates = zip(*val_data)


X = []
count = 0
max_length = 0
for i in range(len(train_sentences)):

  if(len(train_labels[i])>max_length):
    max_length = len(train_labels[i])

  count+=1
  X.append(len(train_sentences[i].split()) == len(train_labels[i]))



class SRLModel(nn.Module):
    def __init__(self, bert_model, lstm_hidden_size, dropout_rate):
        super(SRLModel, self).__init__()
        self.bert =bert_model
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_features=lstm_hidden_size * 2, out_features=1)

    def forward(self, input_ids, predicate_idx, labels=None):
        bert_output = self.bert(input_ids=input_ids)
        sequence_output = bert_output.last_hidden_state

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        logits = self.fc(lstm_output)

        if labels is not None:
            labels_mask = (labels != -100)
            labels = labels[labels_mask]
            logits = logits[labels_mask]

        return logits, labels


class SRLindModel(nn.Module):
    def __init__(self, bert_model, lstm_hidden_size, dropout_rate, predicate_emb_dim):
        super(SRLindModel, self).__init__()
        self.bert = bert_model

        self.predicate_embedding = nn.Embedding(2, predicate_emb_dim)

        self.downstream = nn.Sequential(
            nn.LSTM(input_size=self.bert.config.hidden_size + predicate_emb_dim,
                    hidden_size=lstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=lstm_hidden_size * 2, out_features=1)
        )

    def forward(self, input_ids, predicate_idx, labels=None, attention_mask=None):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        # Create predicate indicator embedding
        predicate_indicator = torch.zeros_like(input_ids)
        predicate_indicator.scatter_(1, predicate_idx.unsqueeze(1), 1)
        predicate_emb = self.predicate_embedding(predicate_indicator)
        
        # Concatenate predicate indicator embeddings with the sequence output
        sequence_output = torch.cat((sequence_output, predicate_emb), dim=-1)

        downstream_output, _ = self.downstream[0](sequence_output)
        downstream_output = self.downstream[1](downstream_output)
        logits = self.downstream[2](downstream_output)

        if labels is not None:
            labels_mask = (labels != -100)
            labels = labels[labels_mask]
            logits = logits[labels_mask]

        return logits, labels

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, predicate_idx, padded_labels = batch
            input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)

            

            logits, labels = model(input_ids, predicate_idx, padded_labels)

            loss = criterion(logits, labels.float().unsqueeze(1))
            total_loss += loss.item()

            mask = labels.ne(-100)
            valid_labels = labels[mask].cpu().numpy()
            valid_logits = logits[mask].cpu().numpy().squeeze()

            all_labels.extend(valid_labels)
            all_logits.extend(valid_logits)

    # Calculate metrics
    #print(all_labels)
    #print(all_logits)
    average_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    all_probs = 1 / (1 + np.exp(-all_logits))

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    f_scores = np.where((precision + recall) != 0.0, (2 *precision * recall)/ (precision + recall + 1e-10), 0)
  
    best_threshold = thresholds[np.argmax(f_scores)]

    # Calculate accuracy and F-score using the best threshold
    preds = (all_probs > best_threshold).astype(int)
    accuracy = accuracy_score(all_labels, preds)
    best_f_score = f1_score(all_labels, preds)

    return average_loss, accuracy, best_f_score, best_threshold

def validate_on_train(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, predicate_idx, padded_labels = batch
            input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)

            logits, labels = model(input_ids, predicate_idx, padded_labels)

            loss = criterion(logits, labels.float().unsqueeze(1))
            total_loss += loss.item()

            mask = labels.ne(-100)
            valid_labels = labels[mask].cpu().numpy()
            valid_logits = logits[mask].cpu().numpy().squeeze()

            all_labels.extend(valid_labels)
            all_logits.extend(valid_logits)

    average_loss = total_loss / len(dataloader)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    all_probs = 1 / (1 + np.exp(-all_logits))

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    f_scores = np.where((precision + recall) != 0.0, (2 * precision * recall) / (precision + recall + 1e-10), 0)

    best_threshold = thresholds[np.argmax(f_scores)]

    preds = (all_probs > best_threshold).astype(int)
    accuracy = accuracy_score(all_labels, preds)
    best_f_score = f1_score(all_labels, preds)

    return average_loss, accuracy, best_f_score, best_threshold

def train_model_old(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value = 1, weighting_method='none', custom_value= 20):

    if weighting_method != 'none':
        negative_count = sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 0])
        positive_count = sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 1])

        if weighting_method == 'direct':
            pos_weight = torch.tensor([negative_count / positive_count], device=device)
        elif weighting_method == 'log':
            pos_weight = torch.tensor([np.log(negative_count / positive_count)], device=device)
        elif weighting_method == 'custom':
            pos_weight = torch.tensor([custom_value], device=device)
        else:
            raise ValueError("Invalid weighting_method value. It must be 'none', 'direct', 'log', or 'custom'.")
    else:
        pos_weight = torch.tensor(1.0, device=device)

    train_accuracies = []
    val_accuracies = []
    train_f_scores = []
    val_f_scores = []
    avg_train_loss_per_epoch = []
    avg_val_loss_per_epoch = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        total_train_loss = 0
        num_train_batches = 0
        for i, batch in enumerate(train_dataloader):
            model.train()
            input_ids, predicate_idx, padded_labels = batch
            input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)

            logits, labels = model(input_ids, predicate_idx, padded_labels)

            criterion.pos_weight = pos_weight

            loss = criterion(logits, labels.float().unsqueeze(1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)

            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += loss.item()
            num_train_batches += 1

            #if i % 3 == 0:
            #    print(f"Batch {i}, Loss: {loss.item()}")

        avg_train_loss_per_epoch.append(total_train_loss / num_train_batches)

        val_loss, val_accuracy, val_f_score, val_threshold = validate(model, val_dataloader, criterion)
        avg_val_loss_per_epoch.append(val_loss)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Best F-score: {val_f_score}, Best Threshold: {val_threshold}")

        if (epoch + 1) % 10 == 0:
            train_loss, train_accuracy, train_f_score, train_threshold = validate_on_train(model, train_dataloader, criterion)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_f_scores.append(train_f_score)
            val_f_scores.append(val_f_score)
            print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Best F-score: {train_f_score}, Best Threshold: {train_threshold}")

    return avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores

model_path =  '/content/drive/MyDrive/nlp_srl'
file_name = 'srl_in_model_1'

def train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value=1, weighting_method='none', custom_value=20, patience=15):
  
  
  model_path =  '/content/drive/MyDrive/nlp_srl'
  #file_name = 'srl_in_model' 

  if weighting_method != 'none':
      negative_count = sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 0])
      positive_count = sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 1])

      if weighting_method == 'direct':
          pos_weight = torch.tensor([negative_count / positive_count], device=device)
      elif weighting_method == 'log':
          pos_weight = torch.tensor([np.log(negative_count / positive_count)], device=device)
      elif weighting_method == 'custom':
          pos_weight = torch.tensor([custom_value], device=device)
      else:
          raise ValueError("Invalid weighting_method value. It must be 'none', 'direct', 'log', or 'custom'.")
  else:
      pos_weight = torch.tensor(1.0, device=device)



  train_accuracies = []
  val_accuracies = []
  train_f_scores = []
  val_f_scores = []
  avg_train_loss_per_epoch = []
  avg_val_loss_per_epoch = []

  # Early stopping initialization
  best_val_accuracy = float('-inf')
  patience_counter = 0

  for epoch in range(num_epochs):
      print(f"Epoch {epoch+1}/{num_epochs}")
      total_train_loss = 0
      num_train_batches = 0

      for i, batch in enumerate(train_dataloader):
          model.train()
          input_ids, predicate_idx, padded_labels = batch
          input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)

          logits, labels = model(input_ids, predicate_idx, padded_labels)

          criterion.pos_weight = pos_weight

          loss = criterion(logits, labels.float().unsqueeze(1))
          loss.backward()

          torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)

          optimizer.step()
          optimizer.zero_grad()

          total_train_loss += loss.item()
          num_train_batches += 1


      avg_train_loss_per_epoch.append(total_train_loss / num_train_batches)

      val_loss, val_accuracy, val_f_score, val_threshold = validate(model, val_dataloader, criterion)
      avg_val_loss_per_epoch.append(val_loss)
      print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Best F-score: {val_f_score}, Best Threshold: {val_threshold}")

      if (epoch + 1) % 10 == 0:
          train_loss, train_accuracy, train_f_score, train_threshold = validate_on_train(model, train_dataloader, criterion)
          train_accuracies.append(train_accuracy)
          val_accuracies.append(val_accuracy)
          train_f_scores.append(train_f_score)
          val_f_scores.append(val_f_score)
          print(f"Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Best F-score: {train_f_score}, Best Threshold: {train_threshold}")

          torch.save(model.state_dict(), f"{model_path}/{file_name+str(epoch)+'.pth'}")
          print("model saved")
        
         #torch.save(model.state_dict(), f"{model_path}/{file_name+str(epoch)+'.pth'}")
          #print("model saved")

      # Early stopping
      if val_accuracy > best_val_accuracy:
          best_val_accuracy = val_accuracy
          patience_counter = 0
      else:
          patience_counter += 1

      if patience_counter >= patience:
          print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation accuracy")
          num_missing_values = num_epochs - epoch - 1
          train_accuracies.extend([None] * num_missing_values)
          val_accuracies.extend([None] * num_missing_values)
          train_f_scores.extend([None] * num_missing_values)
          val_f_scores.extend([None] * num_missing_values)

          #return avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores
          return avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores
          

  return avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores



hyper_parameter_dict = {'learning_rate': [1e-4,1e-5,2e-4,2e-5], 'clip_grad_value':[0.8,1.0,1.5,2.0], 'lstm_hidden_size':[50,70,80,96,128], 'dropout_rate':[0.1,0.2,0.3], 
                        'custom_weight_value': [10, 15, 20 ,27, 40] }

bert_model = BertModel.from_pretrained("bert-base-uncased")

num_labels = 1
hidden_size = 768

max_length = 128
train_dataset = SRLDataset(train_sentences, train_predicates, train_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

max_length = 128
val_dataset = SRLDataset(val_sentences, val_predicates, val_labels, tokenizer, max_length)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

negative_count = sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 0])
print(negative_count)
positive_count = sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 1])
print(positive_count)
pos_weight = torch.tensor([negative_count / positive_count ], device=device)
print(pos_weight)



import csv
import ast

num_epochs = 100

criterion = nn.BCEWithLogitsLoss()

def grid_search(hyper_parameter_dict, results_csv_path):
    # Create the CSV file and write the header
    with open(results_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['learning_rate', 'clip_grad_value', 'lstm_hidden_size', 'dropout_rate', 'custom_weight_value', 'train_accuracies', 'train_f_scores', 'val_accuracies', 'val_f_scores']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for config in itertools.product(*hyper_parameter_dict.values()):
        print(f"Training with hyperparameter configuration: {config}")
        learning_rate, clip_grad_value, lstm_hidden_size, dropout_rate, custom_weight_value = config

        # Train the model with the current configuration of hyperparameters
        model = SRLModel(bert_model, hidden_size, num_labels, lstm_hidden_size=lstm_hidden_size, dropout_rate=dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 15)

        # Save the current configuration and its results to the CSV file
        results_dict = {'learning_rate': learning_rate, 'clip_grad_value': clip_grad_value, 'lstm_hidden_size': lstm_hidden_size, 'dropout_rate': dropout_rate, 'custom_weight_value': custom_weight_value, 'train_accuracies': str(train_accuracies), 'train_f_scores': str(train_f_scores), 'val_accuracies': str(val_accuracies), 'val_f_scores': str(val_f_scores)}
        
        with open(results_csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(results_dict)

results_csv_path = 'grid_search_results_sequential_new.csv'
#grid_search(hyper_parameter_dict, results_csv_path)

# train single
bs = 8
lstm_hidden_size= 35
dropout_rate = 0.2
learning_rate = 3e-5
clip_grad_value = 1.0
custom_weight_value = 27.0

model = SRLindModel(bert_model = bert_model, lstm_hidden_size =lstm_hidden_size, dropout_rate = dropout_rate, predicate_emb_dim = 50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

# train single 
# bs 8
lstm_hidden_size= 35
dropout_rate = 0.2
learning_rate = 3e-5
clip_grad_value = 1.0
custom_weight_value = 27.0

model = SRLindModel(bert_model = bert_model, lstm_hidden_size =lstm_hidden_size, dropout_rate = dropout_rate, predicate_emb_dim = 50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

#model = SRLModel(bert_model,  lstm_hidden_size=lstm_hidden_size, dropout_rate=dropout_rate).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#criterion = nn.BCEWithLogitsLoss()

#bert_model, lstm_hidden_size, dropout_rate, predicate_emb_dim
bs = 8
lstm_hidden_size= 40
dropout_rate = 0.2
learning_rate = 3e-5
clip_grad_value = 1.0
custom_weight_value = 27.0
model = SRLindModel(bert_model = bert_model, lstm_hidden_size =lstm_hidden_size, dropout_rate = dropout_rate, predicate_emb_dim = 50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# train single 
# bs 4
#lstm_hidden_size= 30
#dropout_rate = 0.2
#learning_rate = 3e-5
#clip_grad_value = 1.0
#custom_weight_value = 27.0

# train single
#bs = 8
#lstm_hidden_size= 40
#dropout_rate = 0.2
#learning_rate = 3e-5
#clip_grad_value = 1.0
#custom_weight_value = 27.0
# pred emb 50

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

# train 
#bs = 32
#lstm_hidden_size= 35
#dropout_rate = 0.2
#learning_rate = 3e-5
#clip_grad_value = 1.0
#custom_weight_value = 27.0
# pred emb 30

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

# train single
#bs = 8
#lstm_hidden_size= 35
#dropout_rate = 0.2
#learning_rate = 3e-5
#clip_grad_value = 1.0
#custom_weight_value = 27.0
# likely ind model 

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

# train single 
bs = 32
lstm_hidden_size= 35
dropout_rate = 0.3
learning_rate = 3e-5
clip_grad_value = 1.2
custom_weight_value = 27.0

model = SRLindModel(bert_model = bert_model, lstm_hidden_size =lstm_hidden_size, dropout_rate = dropout_rate, predicate_emb_dim = 50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

# train single
#bs = 8
#lstm_hidden_size= 75
#dropout_rate = 0.3
#learning_rate = 3e-5
#clip_grad_value = 1.0
#custom_weight_value = 27.0

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

# train single
#bs = 64
#lstm_hidden_size= 80
#dropout_rate = 0.2
#learning_rate = 3e-5
#clip_grad_value = 1.5
#custom_weight_value = 27.0

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

#grid_search(hyper_parameter_dict, results_csv_path)

# train single
# bs 20 
#lstm_hidden_size= 70
#dropout_rate = 0.2
#learning_rate = 4e-5
#clip_grad_value = 2.0
#custom_weight_value = 27.0

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 50)

#grid_search_results = grid_search(hyper_parameter_dict, train_dataset, train_dataloader, val_dataloader, criterion, num_epochs)