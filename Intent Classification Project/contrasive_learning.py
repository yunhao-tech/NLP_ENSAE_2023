from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaModel, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataLoader(name_dataset='dyda_da', batch_size=32):
  dataset = load_dataset('silicone', name_dataset, split='train')
  dataset_val = load_dataset('silicone', name_dataset, split='validation')
  dataset_test = load_dataset('silicone', name_dataset, split='test')
  training_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
  val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
  test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
  return training_loader, val_loader, test_loader

def get_labels(name_dataset):
  if name_dataset == 'dyda_da':
    return ["commissive", "directive", "inform", "question"]
  if name_dataset == 'dyda_e':
    return ["anger", "disgust", "fear", "happiness", "no emotion", "sadness", "surprise"]
  if name_dataset == 'meld_e':
    return ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
  if name_dataset == 'meld_s' or name_dataset == 'sem':
    return ["negative", "neutral", "positive"]
  
def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.softmax(logits, dim=dim))
    return -neg_ce.mean()
    
def clip_loss(similarity):
    utterance_loss = contrastive_loss(similarity, dim=0)
    label_loss = contrastive_loss(similarity, dim=1)
    return (utterance_loss + label_loss) / 2.0

def metrics(similarity):
    y = torch.arange(len(similarity)).to(device)
    lab2utt_match_idx = similarity.argmax(dim=0)
    utt2lab_match_idx = similarity.argmax(dim=1)
    utt_acc = (lab2utt_match_idx == y).float().mean()
    lab_acc = (utt2lab_match_idx == y).float().mean()
    return utt_acc, lab_acc

def fine_tune_prediction(model, tokenizer, optimizer, epochs, name_dataset='dyda_da'):
  train_loader, val_loader, test_loader = get_dataLoader(name_dataset=name_dataset)
  for epoch in range(1, epochs + 1):
    model.train()
    losses = []
    acc_lab = []
    acc_utt = []
    with tqdm(train_loader, unit="batch", position=0, leave=True) as tepoch: # evaluate every batch
      for data in tepoch:
        tepoch.set_description(f"Epoch {epoch}") # custome the printed message in tqdm
        optimizer.zero_grad()
        input = tokenizer(data['Utterance'], return_tensors="pt", padding='max_length', max_length=30, truncation=True).to(device)
        label = tokenizer(data['Dialogue_Act'], return_tensors="pt", padding='max_length', max_length=6, truncation=True).to(device)
        utt_embed = model.deberta(**input).last_hidden_state[:,0]
        lab_embed = model.deberta(**label).last_hidden_state[:,0]
        similarity = F.normalize(utt_embed) @ F.normalize(lab_embed).t()
        loss = clip_loss(similarity)
        loss.backward() # backward propagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding gradient 
        optimizer.step() 
        utt_acc, lab_acc = metrics(similarity)
        acc_lab.append(lab_acc.item())
        acc_utt.append(utt_acc.item())
        losses.append(loss.item())
        tepoch.set_postfix(loss=sum(losses)/len(losses), utt_accuracy=100. * sum(acc_utt)/len(acc_utt), lab_accuracy=100. * sum(acc_lab)/len(acc_lab))
    val_acc = evaluate_acc(model, val_loader, tokenizer, name_dataset)
    print(f"Validation: Utterance classification Accuracy: {100 * val_acc :3.2f}%")
  print("Training finished...")
  test_acc = evaluate_acc(model, test_loader, tokenizer, name_dataset)
  print(f"Validation: Utterance classification Accuracy: {100 * test_acc :3.2f}%")
  

def evaluate_acc(model, loader, tokenizer, name_dataset):
  model.eval()
  correct = 0
  total = 0
  class_label = get_labels(name_dataset)
  label = tokenizer(class_label, return_tensors="pt", padding='max_length', max_length=6, truncation=True).to(device)
  lab_embed = model.deberta(**label).last_hidden_state[:,0]
  for data in loader:
    input = tokenizer(data['Utterance'], return_tensors="pt", padding='max_length', max_length=30, truncation=True).to(device)
    utt_embed = model.deberta(**input).last_hidden_state[:,0]
    similarity = F.normalize(utt_embed) @ F.normalize(lab_embed).t()
    pred = torch.argmax(similarity, axis=1)
    correct += sum(data['Label'].to(device) == pred).item()
    total += len(data['Utterance'])
  return correct/total

if __name__ == "__main__":
   
    tokenizer = AutoTokenizer.from_pretrained("diwank/silicone-deberta-pair")
    model = AutoModelForSequenceClassification.from_pretrained("diwank/silicone-deberta-pair").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    fine_tune_prediction(model, tokenizer, optimizer, epochs=3, name_dataset='dyda_da')
