## use pip install on anaconda prompt for the imports

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
import csv
from torch.utils.data import random_split

def map_pos_tags(tag):
    pos_groups = {
        "NN": ["NNS", "NNP", "NNPS"],
        "VB": ["VBD", "VBG", "VBN", "VBP", "VBZ"],
        "CC": ["CC"],
        "DT": ["DT"],
        "JJ": ["JJ", "JJR", "JJS"],
        "IN": ["IN"],
        "PRP": ["PRP", "PRP$"],

        #'other', 'JJ 5', 'IN 6', 'PRP ', 'DT 4', 'NN 1', 'VB 2', 'CC 3'
    }

    for key, value in pos_groups.items():
        if tag in value:
            return key
    return "other"

def load_and_preprocess_data(file_path):
    sentences = []
    predicate_indices = []
    labels = []
    pos_tags = []
    bio_tags = []
    directed_distances = []
    skipped_count = 0
    taken_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        current_sentence = []
        current_predicate_index = None
        current_labels = []
        current_pos_tags = []
        current_bio_tags = []
        current_directed_distances = []
        has_arg1 = False

        for row in reader:
            if row:
                #print(row)
                word, pos, bio, word_idx, sentence_idx = row[:5]
                pos = map_pos_tags(pos)

                if len(row)>5:
                    if row[5] == "PRED":
                           current_predicate_index = int(word_idx)

                current_sentence.append(word)
                current_pos_tags.append(pos)
                current_bio_tags.append(bio)
                label = 1 if "ARG1" in row else 0
                current_labels.append(label)

                if current_predicate_index is not None:
                   #print("here in pred idx not none")
                   current_directed_distances.append(int(word_idx) - current_predicate_index)

                if "ARG1" in row:
                    has_arg1 = True
                    #print("here in has arg1", has_arg1)
            else:
                if current_sentence and has_arg1 and current_predicate_index is not None:
                   # print("here in else of if row")
                    sentences.append(" ".join(current_sentence))
                    predicate_indices.append(current_predicate_index)
                    labels.append(current_labels)
                    pos_tags.append(current_pos_tags)
                    bio_tags.append(current_bio_tags)
                    for i in range(current_predicate_index ,0, -1):
                      current_directed_distances.insert(0,i - current_predicate_index - 1)

                    directed_distances.append(current_directed_distances)

                    taken_count+=1
                else:
                    #print("Skipped sentence:", current_sentence)
                    skipped_count+=1
                    #break

                current_sentence = []
                current_predicate_index = None
                current_labels = []
                current_pos_tags = []
                current_bio_tags = []
                current_directed_distances = []
                has_arg1 = False

    return sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances, skipped_count, taken_count



def load_and_preprocess_data_multiple_files(file_paths):
    sentences = []
    predicate_indices = []
    labels = []
    pos_tags = []
    bio_tags = []
    directed_distances = []
    skipped_counts = []
    taken_counts = []

    for file_path in file_paths:
        result = load_and_preprocess_data(file_path)
        sentences.extend(result[0])
        predicate_indices.extend(result[1])
        labels.extend(result[2])
        pos_tags.extend(result[3])
        bio_tags.extend(result[4])
        directed_distances.extend(result[5])
        skipped_counts.append(result[6])
        taken_counts.append(result[7])

    return sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances, skipped_counts, taken_counts

file_paths = ['partitive_group_nombank.clean.train', 'partitive_group_nombank.clean.test', 'partitive_group_nombank.clean.dev']

sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances, skipped_counts, taken_counts = load_and_preprocess_data_multiple_files(file_paths)

def tokenize_loaded_data(sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances):
    # Create unique dictionaries for pos_tags and bio_tags
    pos_tag_dict = {tag: idx  for idx, tag in enumerate(set(tag for tags in pos_tags for tag in tags))}
    bio_tag_dict = {tag: idx  for idx, tag in enumerate(set(tag for tags in bio_tags for tag in tags))}

    # Convert pos_tags and bio_tags to integers using the dictionaries
    pos_tags = [[pos_tag_dict[tag] for tag in tags] for tags in pos_tags]
    bio_tags = [[bio_tag_dict[tag] for tag in tags] for tags in bio_tags]

    return sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances

def find_num_tags(tags):

  tags_flat = [tag for tag_list in tags for tag in tag_list]
  #print(tags_flat[2])
  #print(len(tags_flat))
  print(set(tags_flat))
  return len(set(tags_flat)) + 1


def pad_and_align_tags(sentence, labels, pos_tags, bio_tags, directed_distances, max_length, tokenizer):
    tokenized_sentence = tokenizer.tokenize(sentence)
    words = sentence.split()

    def align_tags(tags):
        aligned_tags = [-100] * len(tokenized_sentence)
        for word, tag, idx in zip(words, tags, range(len(words))):
            subwords = tokenizer.tokenize(word)
            subword_idx = tokenized_sentence.index(subwords[0], idx)
            aligned_tags[subword_idx] = tag
        return aligned_tags

    aligned_labels = align_tags(labels)
    aligned_pos_tags = align_tags(pos_tags)
    aligned_bio_tags = align_tags(bio_tags)
    aligned_directed_distances = align_tags(directed_distances)

    def pad_tags(tags):
        padded_tags = [-100] + tags[:max_length - 2] + [-100]
        padded_tags = padded_tags + [-100] * (max_length - len(padded_tags))
        return padded_tags

    padded_labels = pad_tags(aligned_labels)
    padded_pos_tags = pad_tags(aligned_pos_tags)
    padded_bio_tags = pad_tags(aligned_bio_tags)
    padded_directed_distances = pad_tags(aligned_directed_distances)

    return padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances


class SRLfeatDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, tokenizer, max_length):
        self.sentences, self.predicate_indices, self.labels, self.pos_tags, self.bio_tags, self.directed_distances, _, _ = load_and_preprocess_data_multiple_files(file_paths)
        self.sentences, self.predicate_indices, self.labels, self.pos_tags, self.bio_tags, self.directed_distances = tokenize_loaded_data(self.sentences, self.predicate_indices, self.labels, self.pos_tags, self.bio_tags, self.directed_distances)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.padded_input_ids = []
        self.padded_labels = []
        self.padded_pos_tags = []
        self.padded_bio_tags = []
        self.padded_directed_distances = []

        for sentence, labels, pos_tags, bio_tags, directed_distances in zip(self.sentences, self.labels, self.pos_tags, self.bio_tags, self.directed_distances):
            input_ids, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = self.process_sentence(sentence, labels, pos_tags, bio_tags, directed_distances)
            self.padded_input_ids.append(input_ids)
            self.padded_labels.append(padded_labels)
            self.padded_pos_tags.append(padded_pos_tags)
            self.padded_bio_tags.append(padded_bio_tags)
            self.padded_directed_distances.append(padded_directed_distances)

    def process_sentence(self, sentence, labels, pos_tags, bio_tags, directed_distances):
        tokenized_sentence = self.tokenizer.tokenize(sentence)
        encoded_sentence = self.tokenizer.encode(sentence, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = torch.tensor(encoded_sentence, dtype=torch.long)

        padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = pad_and_align_tags(sentence, labels, pos_tags, bio_tags, directed_distances, self.max_length, self.tokenizer)

        padded_labels = torch.tensor(padded_labels, dtype=torch.long)
        padded_pos_tags = torch.tensor(padded_pos_tags, dtype=torch.long)
        padded_bio_tags = torch.tensor(padded_bio_tags, dtype=torch.long)
        padded_directed_distances = torch.tensor(padded_directed_distances, dtype=torch.float).unsqueeze(-1)

        return input_ids, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances

    def __getitem__(self, index):
        input_ids = self.padded_input_ids[index]
        predicate_idx = self.predicate_indices[index]
        padded_labels = self.padded_labels[index]
        padded_pos_tags = self.padded_pos_tags[index]
        padded_bio_tags = self.padded_bio_tags[index]
        padded_directed_distances = self.padded_directed_distances[index]

        return input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances

    def __len__(self):
        return len(self.sentences)

file_paths = ['partitive_group_nombank.clean.train', 'partitive_group_nombank.clean.test', 'partitive_group_nombank.clean.dev']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_length = 128

srl_feat_dataset = SRLfeatDataset(file_paths, tokenizer, max_length)


# Calculate the number of samples for the train and validation sets
total_samples = len(srl_feat_dataset)
train_samples = int(total_samples * 0.8)
val_samples = total_samples - train_samples

# Split the dataset into train and validation sets
train_dataset, val_dataset = random_split(srl_feat_dataset, [train_samples, val_samples])

batch_size = 4  # Choose a batch size according to your needs

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = next(iter(train_loader))

max_bio = 0
max_pos = 0
for batch in train_loader:

            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
            max_pos_tags_batch = torch.max(padded_pos_tags)
            max_bio_tags_batch = torch.max(padded_bio_tags)
            if max_pos_tags_batch> max_pos:
              max_pos = max_pos_tags_batch
            if max_bio_tags_batch> max_bio:
              max_bio = max_bio_tags_batch



class SRLfeatModel(nn.Module):

    def __init__(self, bert_model, pos_tag_embedding_dim, bio_tag_embedding_dim, num_pos_tags, num_bio_tags, linear_output_dim, num_labels, dropout_rate):
        super(SRLfeatModel, self).__init__()

        self.bert = bert_model
        self.pos_tag_embedder = nn.Embedding(num_embeddings=num_pos_tags + 1, embedding_dim=pos_tag_embedding_dim)
        self.bio_tag_embedder = nn.Embedding(num_embeddings=num_bio_tags + 1, embedding_dim=bio_tag_embedding_dim)

        self.linear1 = nn.Linear(self.bert.config.hidden_size, linear_output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(linear_output_dim + pos_tag_embedding_dim + bio_tag_embedding_dim + 1, num_labels)

    def forward(self, input_ids, pos_tags, bio_tags, directed_distances, padded_labels):
        bert_output = self.bert(input_ids)
        sequence_output = bert_output.last_hidden_state

        transformed_bert_output = self.linear1(sequence_output)
        transformed_bert_output = self.dropout(transformed_bert_output)

        # Mask and replace -100 values in pos_tags and bio_tags with 0
        pos_tags = pos_tags.masked_fill(pos_tags == -100, 8)
        bio_tags = bio_tags.masked_fill(bio_tags == -100, 17)

        pos_tag_embeddings = self.pos_tag_embedder(pos_tags)
        bio_tag_embeddings = self.bio_tag_embedder(bio_tags)

        concatenated_features = torch.cat([transformed_bert_output, pos_tag_embeddings, bio_tag_embeddings, directed_distances], dim=-1)

        logits = self.linear2(concatenated_features)

        mask = (padded_labels != -100)
        masked_labels = torch.masked_select(padded_labels, mask)
        masked_logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, logits.shape[-1])

        return masked_logits, masked_labels

bert_model = BertModel.from_pretrained("bert-base-uncased")

num_bio_tags = 18
num_pos_tags = 9
num_labels = 1
linear_output_dim = 256
pos_tag_embedding_dim = 8
bio_tag_embedding_dim = 16
dropout_rate = 0.1

model = SRLfeatModel(bert_model, pos_tag_embedding_dim, bio_tag_embedding_dim, num_pos_tags, num_bio_tags, linear_output_dim, num_labels, dropout_rate)

input_ids_batch = input_ids
padded_pos_tags_batch = padded_pos_tags
padded_bio_tags_batch = padded_bio_tags
padded_directed_distances_batch = padded_directed_distances
padded_labels_batch = padded_labels

print("input_ids_batch shape:", input_ids_batch.shape)
print("padded_pos_tags_batch shape:", padded_pos_tags_batch.shape)
print("padded_bio_tags_batch shape:", padded_bio_tags_batch.shape)
print("padded_directed_distances_batch shape:", padded_directed_distances_batch.shape)
print("padded_labels_batch shape:", padded_labels_batch.shape)

#padded_directed_distances_batch = padded_directed_distances_batch.squeeze(-1)

print("padded_directed_distances_batch shape:", padded_directed_distances_batch.shape)

masked_logits, masked_labels = model(input_ids_batch, padded_pos_tags_batch, padded_bio_tags_batch, padded_directed_distances_batch, padded_labels_batch)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = model.to(device)

def validate_on_train(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags,  padded_directed_distances  = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device), padded_pos_tags.to(device),padded_bio_tags.to(device),  padded_directed_distances.to(device)

            logits, labels = model( input_ids, padded_pos_tags, padded_bio_tags, padded_directed_distances, padded_labels)

            #print(logits.shape)
            #print(labels.shape)
            #print(labels.dtype)
            #labels= labels.squeeze(1).float()
            #labels = labels.to(dtype=torch.float32)
            #labels = labels.float()
            #print(labels.dtype)

            #print(logits.shape)

            #labels = labels.squeeze(1)
            #print(labels.dtype)
            #print(labels.shape)

            #labels = labels.to(dtype=torch.float32)
            #print(labels)


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

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags,  padded_directed_distances  = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device), padded_pos_tags.to(device),padded_bio_tags.to(device),  padded_directed_distances.to(device)

            logits, labels = model( input_ids, padded_pos_tags, padded_bio_tags, padded_directed_distances, padded_labels)

            #print(logits.shape)
            #print(labels.shape)
            #print(labels.dtype)
            #labels= labels.squeeze(1).float()
            #labels = labels.to(dtype=torch.float32)
            #labels = labels.float()
            #print(labels.dtype)

            #print(logits.shape)

            #labels = labels.squeeze(1)
            #print(labels.dtype)
            #print(labels.shape)

            #labels = labels.to(dtype=torch.float32)
            #print(labels)


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

#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

def train_model(model, train_dataset, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, clip_grad_value=1, weighting_method='none', custom_value=20, patience=15):

  if weighting_method != 'none':
      negative_count = 27.0 #sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 0])
      positive_count = 1.0 #sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 1])

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
          #input_ids, predicate_idx, padded_labels = batch
          #input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)

          #logits, labels = model(input_ids, predicate_idx, padded_labels)
          input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
          input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags,  padded_directed_distances  = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device), padded_pos_tags.to(device),padded_bio_tags.to(device),  padded_directed_distances.to(device)

          logits, labels = model( input_ids, padded_pos_tags, padded_bio_tags, padded_directed_distances, padded_labels)

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

num_bio_tags = 18
num_pos_tags = 9
num_labels = 1
linear_output_dim = 256
pos_tag_embedding_dim = 8
bio_tag_embedding_dim = 16
dropout_rate = 0.2
num_epochs = 100
learning_rate = 3e-5
clip_grad_value = 1.0
custom_weight_value = 27.0

model = SRLfeatModel(bert_model, pos_tag_embedding_dim, bio_tag_embedding_dim, num_pos_tags, num_bio_tags, linear_output_dim, num_labels, dropout_rate).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# train single
#bs = 8
#lstm_hidden_size= 40
#dropout_rate = 0.2
#learning_rate = 3e-5
#clip_grad_value = 1.0
#custom_weight_value = 27.0
# pred emb 50

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_loader, val_loader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 30)

## freezing doesnt seem to work

class SRLfeatModel(nn.Module):

    def __init__(self, bert_model, pos_tag_embedding_dim, bio_tag_embedding_dim, num_pos_tags, num_bio_tags, linear_output_dim, num_labels, dropout_rate):
        super(SRLfeatModel, self).__init__()

        self.bert = bert_model
                # Freeze BERT model's weights

        self.pos_tag_embedder = nn.Embedding(num_embeddings=num_pos_tags + 1, embedding_dim=pos_tag_embedding_dim)
        self.bio_tag_embedder = nn.Embedding(num_embeddings=num_bio_tags + 1, embedding_dim=bio_tag_embedding_dim)

        self.linear1 = nn.Linear(self.bert.config.hidden_size, linear_output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(linear_output_dim + pos_tag_embedding_dim + bio_tag_embedding_dim + 1, num_labels)

    def forward(self, input_ids, pos_tags, bio_tags, directed_distances, padded_labels):
        bert_output = self.bert(input_ids)
        sequence_output = bert_output.last_hidden_state

        transformed_bert_output = self.linear1(sequence_output)
        transformed_bert_output = self.dropout(transformed_bert_output)

        # Mask and replace -100 values in pos_tags and bio_tags with 0
        pos_tags = pos_tags.masked_fill(pos_tags == -100, 8)
        bio_tags = bio_tags.masked_fill(bio_tags == -100, 17)

        pos_tag_embeddings = self.pos_tag_embedder(pos_tags)
        bio_tag_embeddings = self.bio_tag_embedder(bio_tags)

        concatenated_features = torch.cat([transformed_bert_output, pos_tag_embeddings, bio_tag_embeddings, directed_distances], dim=-1)

        logits = self.linear2(concatenated_features)

        mask = (padded_labels != -100)
        masked_labels = torch.masked_select(padded_labels, mask)
        masked_logits = torch.masked_select(logits, mask.unsqueeze(-1)).view(-1, logits.shape[-1])

        return masked_logits, masked_labels

model = SRLfeatModel(bert_model, pos_tag_embedding_dim, bio_tag_embedding_dim, num_pos_tags, num_bio_tags, linear_output_dim, num_labels, dropout_rate).to(device)

# train single
#bs = 8
#lstm_hidden_size= 40
#dropout_rate = 0.2
#learning_rate = 3e-5
#clip_grad_value = 1.0
#custom_weight_value = 27.0
# pred emb 50

avg_train_loss_per_epoch, avg_val_loss_per_epoch, train_accuracies, val_accuracies, train_f_scores, val_f_scores = train_model(model, train_dataset, train_loader, val_loader, criterion, optimizer, num_epochs, clip_grad_value, weighting_method='custom', custom_value=custom_weight_value, patience = 20)
