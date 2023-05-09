## this files sets up a dataset for multi task learning.
## The auxilary task being driven by linguistic features.

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
        "DT": ["DT"],
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

def tokenize_loaded_data(sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances):
    # Create unique dictionaries for pos_tags and bio_tags
    pos_tag_dict = {tag: idx for idx, tag in enumerate(set(tag for tags in pos_tags for tag in tags))}
    bio_tag_dict = {tag: idx for idx, tag in enumerate(set(tag for tags in bio_tags for tag in tags))}

    # Convert pos_tags and bio_tags to integers using the dictionaries
    pos_tags = [[pos_tag_dict[tag] for tag in tags] for tags in pos_tags]
    bio_tags = [[bio_tag_dict[tag] for tag in tags] for tags in bio_tags]

    return sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances

file_paths = ['partitive_group_nombank.clean.train', 'partitive_group_nombank.clean.test', 'partitive_group_nombank.clean.dev']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def pad_sequences(sequences, maxlen, padding='pre', truncating='pre', value=-100):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) > maxlen:
            if truncating == 'pre':
                sequence = sequence[-maxlen:]
            elif truncating == 'post':
                sequence = sequence[:maxlen]
        elif len(sequence) < maxlen:
            if padding == 'pre':
                sequence = [value] * (maxlen - len(sequence)) + sequence
            elif padding == 'post':
                sequence = sequence + [value] * (maxlen - len(sequence))
        padded_sequences.append(sequence)
    return padded_sequences

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

class SRLauxDataset(torch.utils.data.Dataset):
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
        padded_directed_distances = torch.tensor(padded_directed_distances, dtype=torch.long)

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

## dataset defined here
max_length = 128
srl_aux_dataset = SRLauxDataset(file_paths, tokenizer, max_length)



input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances  = srl_aux_dataset.__getitem__(0)

srl_aux_dataset.labels[0]

len(srl_aux_dataset)

sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances, _, _ = load_and_preprocess_data_multiple_files(file_paths)
sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances = tokenize_loaded_data(sentences, predicate_indices, labels, pos_tags, bio_tags, directed_distances)

pos_tags[0]

bio_tags[0]

def find_num_tags(tags):

  tags_flat = [tag for tag_list in tags for tag in tag_list]
  #print(tags_flat[2])
  #print(len(tags_flat))
  return len(set(tags_flat)) + 1

num_pos_tags = find_num_tags(pos_tags)

pos_tags[1]

num_pos_tags

num_bio_tags = find_num_tags(bio_tags)

num_bio_tags


# use next split
# Calculate the number of samples for the train and validation sets
#train_samples = 10000
#val_samples_first = 1000
#val_samples_last = 500

# Create the train and validation sets using list slicing
#train_dataset = Subset(srl_aux_dataset, range(0, train_samples))
#val_dataset = Subset(srl_aux_dataset, list(range(train_samples, train_samples + val_samples_first)) + list(range(-val_samples_last, 0)))


# Calculate the number of samples for the train and validation sets
total_samples = len(srl_aux_dataset)
train_samples = int(total_samples * 0.8)
val_samples = total_samples - train_samples

# Split the dataset into train and validation sets
train_dataset, val_dataset = random_split(srl_aux_dataset, [train_samples, val_samples])

batch_size = 4  # Choose a batch size according to your needs

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = next(iter(train_loader))
input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device), padded_pos_tags.to(device), padded_bio_tags.to(device), padded_directed_distances.to(device)

class SRLauxModel(nn.Module):
    def __init__(self, bert_model, lstm_hidden_size, dropout_rate, layers_to_use=[1, 2, 3], num_pos_tags=9, num_bio_tags=18):
        super(SRLauxModel, self).__init__()
        self.bert = bert_model
        self.layers_to_use = layers_to_use
        self.layer_weights = nn.Parameter(torch.rand(len(layers_to_use), dtype=torch.float))
        self.softmax = nn.Softmax(dim=0)

        self.auxiliary_pos = nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_pos_tags)
        self.auxiliary_bio = nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_bio_tags)
        self.auxiliary_directed_distance = nn.Linear(in_features=self.bert.config.hidden_size, out_features=1)

        self.downstream = nn.Sequential(
            nn.LSTM(input_size=self.bert.config.hidden_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=lstm_hidden_size * 2, out_features=1)
        )

    def forward(self, input_ids, predicate_idx, labels=None):
        input_embeddings = self.bert.embeddings(input_ids)

        # Create predicate indicator embedding
        predicate_mask = torch.zeros_like(input_ids).scatter_(1, predicate_idx.view(-1, 1), 1)
        predicate_indicator = self.bert.embeddings.token_type_embeddings(predicate_mask.to(input_ids.device))

        # Add predicate indicator to input embeddings
        input_embeddings = input_embeddings + predicate_indicator

        bert_output = self.bert(inputs_embeds=input_embeddings, output_hidden_states=True)
        all_layer_outputs = bert_output.hidden_states

        selected_layer_outputs = [all_layer_outputs[i] for i in self.layers_to_use]
        weighted_outputs = [self.softmax(self.layer_weights)[i] * output for i, output in enumerate(selected_layer_outputs)]
        weighted_average = torch.stack(weighted_outputs).sum(dim=0)

        pos_logits = self.auxiliary_pos(weighted_average)
        bio_logits = self.auxiliary_bio(weighted_average)
        directed_distance_logits = self.auxiliary_directed_distance(weighted_average)

        downstream_output, _ = self.downstream[0](weighted_average)
        downstream_output = self.downstream[1](downstream_output)
        logits = self.downstream[2](downstream_output)

        if labels is not None:
            labels_mask = (labels != -100)
            labels = labels[labels_mask]
            logits = logits[labels_mask]

       # logits = logits.squeeze(-1)

        return logits, labels, pos_logits, bio_logits, directed_distance_logits

bert_model = BertModel.from_pretrained("bert-base-uncased")

model = SRLauxModel(bert_model = bert_model , lstm_hidden_size = 50, dropout_rate = 0.2, layers_to_use=[1, 2, 3], num_pos_tags=9, num_bio_tags=18).to(device)

logits, labels, pos_logits, bio_logits, directed_distance_logits =model(input_ids = input_ids, predicate_idx = predicate_idx, labels=padded_labels)

labels.shape

labels

#logits

logits.shape

directed_distance_logits.shape

bio_logits.shape

pos_logits.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def validate_on_train(model, dataloader):
    main_task_criterion = nn.BCEWithLogitsLoss()
    pos_tag_criterion = nn.CrossEntropyLoss()
    bio_tag_criterion = nn.CrossEntropyLoss()
    directed_distance_criterion = nn.MSELoss()

    model.eval()
    total_loss = 0
    total_aux_pos_loss = 0
    total_aux_bio_loss = 0
    total_aux_directed_distance_loss = 0

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
            input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)
            padded_pos_tags, padded_bio_tags, padded_directed_distances = padded_pos_tags.to(device), padded_bio_tags.to(device), padded_directed_distances.to(device)

            logits, labels, pos_logits, bio_logits, directed_distance_logits = model(input_ids, predicate_idx, padded_labels)

            loss = main_task_criterion(logits, labels.float().unsqueeze(1))
            aux_pos_loss = pos_tag_criterion(pos_logits.view(-1, pos_logits.shape[-1]), padded_pos_tags.view(-1))
            aux_bio_loss = bio_tag_criterion(bio_logits.view(-1, bio_logits.shape[-1]), padded_bio_tags.view(-1))
            aux_directed_distance_loss = directed_distance_criterion(directed_distance_logits.squeeze(-1), padded_directed_distances.float())

            total_loss += loss.item()
            total_aux_pos_loss += aux_pos_loss.item()
            total_aux_bio_loss += aux_bio_loss.item()
            total_aux_directed_distance_loss += aux_directed_distance_loss.item()

            mask = labels.ne(-100)
            valid_labels = labels[mask].cpu().numpy()
            valid_logits = logits[mask].cpu().numpy().squeeze()

            all_labels.extend(valid_labels)
            all_logits.extend(valid_logits)

        # Masking the -100 labels for auxiliary tasks
        aux_mask = padded_labels.ne(-100).view(-1)

        average_loss = total_loss / len(dataloader)
        average_aux_pos_loss = (total_aux_pos_loss / len(dataloader)) * aux_mask.float().mean()
        average_aux_bio_loss = (total_aux_bio_loss / len(dataloader)) * aux_mask.float().mean()
        average_aux_directed_distance_loss = (total_aux_directed_distance_loss / len(dataloader)) * aux_mask.float().mean()

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    all_probs = 1 / (1 + np.exp(-all_logits))

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    f_scores = np.where((precision + recall) != 0.0, (2 * precision * recall) / (precision + recall + 1e-10), 0)

    best_threshold = thresholds[np.argmax(f_scores)]

    preds = (all_probs > best_threshold).astype(int)
    accuracy = accuracy_score(all_labels, preds)
    best_f_score = f1_score(all_labels, preds)

    return average_loss, accuracy, best_f_score, best_threshold, average_aux_pos_loss, average_aux_bio_loss, average_aux_directed_distance_loss

def validate(model, dataloader):
    main_task_criterion = nn.BCEWithLogitsLoss()
    pos_tag_criterion = nn.CrossEntropyLoss()
    bio_tag_criterion = nn.CrossEntropyLoss()
    directed_distance_criterion = nn.MSELoss()

    model.eval()
    total_loss = 0
    total_aux_pos_loss = 0
    total_aux_bio_loss = 0
    total_aux_directed_distance_loss = 0

    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
            input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)
            padded_pos_tags, padded_bio_tags, padded_directed_distances = padded_pos_tags.to(device), padded_bio_tags.to(device), padded_directed_distances.to(device)

            logits, labels, pos_logits, bio_logits, directed_distance_logits = model(input_ids, predicate_idx, padded_labels)

            loss = main_task_criterion(logits, labels.float().unsqueeze(1))
            aux_pos_loss = pos_tag_criterion(pos_logits.view(-1, pos_logits.shape[-1]), padded_pos_tags.view(-1))
            aux_bio_loss = bio_tag_criterion(bio_logits.view(-1, bio_logits.shape[-1]), padded_bio_tags.view(-1))
            aux_directed_distance_loss = directed_distance_criterion(directed_distance_logits.squeeze(-1), padded_directed_distances.float())

            total_loss += loss.item()
            total_aux_pos_loss += aux_pos_loss.item()
            total_aux_bio_loss += aux_bio_loss.item()
            total_aux_directed_distance_loss += aux_directed_distance_loss.item()

            mask = labels.ne(-100)
            valid_labels = labels[mask].cpu().numpy()
            valid_logits = logits[mask].cpu().numpy().squeeze()

            all_labels.extend(valid_labels)
            all_logits.extend(valid_logits)

        # Masking the -100 labels for auxiliary tasks
        aux_mask = padded_labels.ne(-100).view(-1)

        average_loss = total_loss / len(dataloader)
        average_aux_pos_loss = (total_aux_pos_loss / len(dataloader)) * aux_mask.float().mean()
        average_aux_bio_loss = (total_aux_bio_loss / len(dataloader)) * aux_mask.float().mean()
        average_aux_directed_distance_loss = (total_aux_directed_distance_loss / len(dataloader)) * aux_mask.float().mean()

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    all_probs = 1 / (1 + np.exp(-all_logits))

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    f_scores = np.where((precision + recall) != 0.0, (2 * precision * recall) / (precision + recall + 1e-10), 0)

    best_threshold = thresholds[np.argmax(f_scores)]

    preds = (all_probs > best_threshold).astype(int)
    accuracy = accuracy_score(all_labels, preds)
    best_f_score = f1_score(all_labels, preds)

    return average_loss, accuracy, best_f_score, best_threshold, average_aux_pos_loss, average_aux_bio_loss, average_aux_directed_distance_loss

model_path =  '/content/drive/MyDrive/nlp_srl'
file_name = 'srl_aux_model_1'

def train_model_aux_old(model, train_dataset, train_dataloader, val_dataloader, optimizer, num_epochs, task_weights, clip_grad_value=1, weighting_method='none', custom_value=20, patience=15):

    main_task_criterion = nn.BCEWithLogitsLoss()
    pos_tag_criterion = nn.CrossEntropyLoss()
    bio_tag_criterion = nn.CrossEntropyLoss()

    # For regression
    directed_distance_criterion = nn.MSELoss()

    if weighting_method != 'none':
        negative_count = 24 #sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 0])
        positive_count = 1 #sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 1])

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
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
            input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)
            padded_pos_tags, padded_bio_tags, padded_directed_distances = padded_pos_tags.to(device), padded_bio_tags.to(device), padded_directed_distances.to(device)

            logits, labels, pos_logits, bio_logits, directed_distance_logits = model(input_ids, predicate_idx, padded_labels)

            main_task_criterion.pos_weight = pos_weight
            main_task_loss = main_task_criterion(logits, labels.float().unsqueeze(1))

            pos_tag_loss = pos_tag_criterion(pos_logits.view(-1, pos_logits.shape[-1]), padded_pos_tags.view(-1))
            bio_tag_loss = bio_tag_criterion(bio_logits.view(-1, bio_logits.shape[-1]), padded_bio_tags.view(-1))
            directed_distance_loss = directed_distance_criterion(directed_distance_logits.view(-1, directed_distance_logits.shape[-1]), padded_directed_distances.view(-1))

           # total_loss = main_task_loss + pos_tag_loss + bio_tag_loss + directed_distance_loss
            total_loss = task_weights[0]*main_task_loss + task_weights[1]*pos_tag_loss + task_weights[2]*bio_tag_loss + task_weights[3]*directed_distance_loss


            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)

            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += total_loss.item()
            num_train_batches += 1

        avg_train_loss_per_epoch.append(total_train_loss / num_train_batches)

        val_loss, val_accuracy, val_f_score, val_threshold, average_aux_pos_loss, average_aux_bio_loss, average_aux_directed_distance_loss = validate(model, val_dataloader)
        avg_val_loss_per_epoch.append(val_loss)

        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Best F-score: {val_f_score}, Best Threshold: {val_threshold}, POS loss: {average_aux_pos_loss}, BIO loss; {average_aux_bio_loss}, Dir dist loss : {average_aux_directed_distance_loss}")

        if (epoch + 1) % 10 == 0:
            average_loss, train_accuracy, train_f_score, train_threshold, average_aux_pos_loss, average_aux_bio_loss, average_aux_directed_distance_loss= validate_on_train(model, train_dataloader)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_f_scores.append(train_f_score)
            val_f_scores.append(val_f_score)

            torch.save(model.state_dict(), f"{model_path}/{file_name+str(epoch)+'.pth'}")
            print("model saved")

            print(f"Train Loss: {val_loss}, Train Accuracy: {val_accuracy}, Best F-score: {train_f_score}, Best Threshold: {train_threshold}, POS loss: {average_aux_pos_loss}, BIO loss; {average_aux_bio_loss}, Dir dist loss : {average_aux_directed_distance_loss}")

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

### new
def train_model_aux(model, train_dataset, train_dataloader, val_dataloader, optimizer, num_epochs, task_weights, clip_grad_value=1, weighting_method='none', custom_value=20, patience=15):

    main_task_criterion = nn.BCEWithLogitsLoss()
    pos_tag_criterion = nn.CrossEntropyLoss()
    bio_tag_criterion = nn.CrossEntropyLoss()

    # For regression
    directed_distance_criterion = nn.MSELoss()

    if weighting_method != 'none':
        negative_count = 24 #sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 0])
        positive_count = 1 #sum([1 for label_seq in train_dataset.labels for label in label_seq if label == 1])

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
            input_ids, predicate_idx, padded_labels, padded_pos_tags, padded_bio_tags, padded_directed_distances = batch
            input_ids, predicate_idx, padded_labels = input_ids.to(device), predicate_idx.to(device), padded_labels.to(device)
            padded_pos_tags, padded_bio_tags, padded_directed_distances = padded_pos_tags.to(device), padded_bio_tags.to(device), padded_directed_distances.to(device)

            logits, labels, pos_logits, bio_logits, directed_distance_logits = model(input_ids, predicate_idx, padded_labels)

            main_task_criterion.pos_weight = pos_weight
            main_task_loss = main_task_criterion(logits, labels.float().unsqueeze(1))

            pos_tag_loss = pos_tag_criterion(pos_logits.view(-1, pos_logits.shape[-1]), padded_pos_tags.view(-1))
            bio_tag_loss = bio_tag_criterion(bio_logits.view(-1, bio_logits.shape[-1]), padded_bio_tags.view(-1))
            padded_directed_distances.unsqueeze(-1)
        #    print("directed_distance_logits",directed_distance_logits.shape)
       #     print("padded_directed_distances",padded_directed_distances.shape)
            directed_distance_loss = directed_distance_criterion(directed_distance_logits.float().view(-1, directed_distance_logits.shape[-1]), padded_directed_distances.float().view(-1,directed_distance_logits.shape[-1]))
         #   print("after")
          #  print(directed_distance_logits.float().view(-1, directed_distance_logits.shape[-1]).shape)
          #  print(padded_directed_distances.float().view(-1).shape)



           # total_loss = main_task_loss + pos_tag_loss + bio_tag_loss + directed_distance_loss
            task_weights = task_weights.float()
            total_loss = task_weights[0]*main_task_loss + task_weights[1]*pos_tag_loss + task_weights[2]*bio_tag_loss + task_weights[3]*directed_distance_loss


            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_value)

            optimizer.step()
            optimizer.zero_grad()

            total_train_loss += total_loss.item()
            num_train_batches += 1

        avg_train_loss_per_epoch.append(total_train_loss / num_train_batches)

        val_loss, val_accuracy, val_f_score, val_threshold, average_aux_pos_loss, average_aux_bio_loss, average_aux_directed_distance_loss = validate(model, val_dataloader)
        avg_val_loss_per_epoch.append(val_loss)

        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Best F-score: {val_f_score}, Best Threshold: {val_threshold}, POS loss: {average_aux_pos_loss}, BIO loss; {average_aux_bio_loss}, Dir dist loss : {average_aux_directed_distance_loss}")

        if (epoch + 1) % 10 == 0:
            average_loss, train_accuracy, train_f_score, train_threshold, average_aux_pos_loss, average_aux_bio_loss, average_aux_directed_distance_loss= validate_on_train(model, train_dataloader)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_f_scores.append(train_f_score)
            val_f_scores.append(val_f_score)

            torch.save(model.state_dict(), f"{model_path}/{file_name+str(epoch)+'.pth'}")
            print("model saved")

            print(f"Train Loss: {val_loss}, Train Accuracy: {val_accuracy}, Best F-score: {train_f_score}, Best Threshold: {train_threshold}, POS loss: {average_aux_pos_loss}, BIO loss; {average_aux_bio_loss}, Dir dist loss : {average_aux_directed_distance_loss}")

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

batch_size = 8  # Choose a batch size according to your needs

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



num_epochs = 100
lstm_hidden_size= 50
dropout_rate = 0.3
learning_rate = 3e-5
clip_grad_value = 1.5
custom_weight_value = 27.0
layers_to_use = [1,2,3]
task_weights = torch.tensor([1.0, 0.1,0.1,0.06]).to(device)

model = SRLauxModel(bert_model = bert_model , lstm_hidden_size = lstm_hidden_size, dropout_rate = dropout_rate, layers_to_use=layers_to_use, num_pos_tags=9, num_bio_tags=18).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

train_model_aux(model = model, train_dataset = train_dataset, train_dataloader=train_dataloader , val_dataloader = val_dataloader,
                optimizer = optimizer, num_epochs = num_epochs, task_weights = task_weights, clip_grad_value=clip_grad_value, weighting_method='custom', custom_value=custom_weight_value , patience=25)

num_epochs = 100
lstm_hidden_size= 50
dropout_rate = 0.3
learning_rate = 3e-5
clip_grad_value = 1.5
custom_weight_value = 27.0
layers_to_use = [1,2,3]
task_weights = torch.tensor([1.0, 0.1,0.1,0.05]).to(device)

model = SRLauxModel(bert_model = bert_model , lstm_hidden_size = lstm_hidden_size, dropout_rate = dropout_rate, layers_to_use=layers_to_use, num_pos_tags=9, num_bio_tags=18).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

train_model_aux(model = model, train_dataset = train_dataset, train_dataloader=train_dataloader , val_dataloader = val_dataloader,
                optimizer = optimizer, num_epochs = num_epochs, task_weights = task_weights, clip_grad_value=clip_grad_value, weighting_method='custom', custom_value=custom_weight_value , patience=25)

batch_size = 4  # Choose a batch size according to your needs

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

num_epochs = 100
lstm_hidden_size= 70
dropout_rate = 0.3
learning_rate = 3e-5
clip_grad_value = 1.5
custom_weight_value = 27.0
layers_to_use = [1,2,3]
task_weights = torch.tensor([1.0, 0.2,0.2,0.2]).to(device)

model = SRLauxModel(bert_model = bert_model , lstm_hidden_size = lstm_hidden_size, dropout_rate = dropout_rate, layers_to_use=layers_to_use, num_pos_tags=9, num_bio_tags=18).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()



train_model_aux(model = model, train_dataset = train_dataset, train_dataloader=train_dataloader , val_dataloader = val_dataloader,
                optimizer = optimizer, num_epochs = num_epochs, task_weights = task_weights, clip_grad_value=clip_grad_value, weighting_method='custom', custom_value=custom_weight_value , patience=25)
