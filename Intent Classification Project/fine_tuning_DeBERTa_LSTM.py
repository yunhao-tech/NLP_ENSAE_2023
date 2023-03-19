from tqdm.auto import tqdm
import numpy as np
from itertools import compress

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaModel, AutoTokenizer
from datasets import load_dataset

class My_Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.id_unique = np.unique(np.array(dataset["Dialogue_ID"]))
        
    def __len__(self):
        return len(set(self.dataset["Dialogue_ID"]))

    def __getitem__(self, index):
        bool_l = np.array(self.dataset["Dialogue_ID"]) == self.id_unique[index]
        ind = list(compress(range(len(bool_l)), bool_l)) # the indices of utterances whose Dialogue ID equals to "index+1"
        data = self.dataset[ind]
        sample = {
            "Utterance": data['Utterance'],
            "Label": torch.tensor(data['Label']),
            }
        return sample


def get_dataLoader(name_dataset='dyda_da'):
  dataset_train = My_Dataset(load_dataset('silicone', name_dataset, split='train'))
  dataset_val = My_Dataset(load_dataset('silicone', name_dataset, split='validation'))
  dataset_test = My_Dataset(load_dataset('silicone', name_dataset, split='test'))
  training_loader = DataLoader(dataset_train, batch_size=1)
  val_loader = DataLoader(dataset_val, batch_size=1)
  test_loader = DataLoader(dataset_test, batch_size=1)
  return training_loader, val_loader, test_loader

class MyModel(torch.nn.Module):
  def __init__(self, trained_transformer, num_class, drop_out=0.15, fine_tune_whole=True):
    super().__init__()
    self.trained_transformer = trained_transformer
    self.lstm = torch.nn.LSTM(input_size=768, hidden_size=384, num_layers=1, bidirectional=True)
    self.linear2 = torch.nn.Linear(768, 768)
    self.linear3 = torch.nn.Linear(768, num_class)
    self.dropout = torch.nn.Dropout(p=drop_out)
    self.relu = torch.nn.ReLU()
    if not fine_tune_whole:
      for param in self.trained_transformer.parameters():
        param.requires_grad = False
  
  def forward(self, inputs):
    x = self.trained_transformer(**inputs).last_hidden_state[:,0,:]
    x, (_, _) = self.lstm(x) # Take the last hidden state 
    x = self.relu(self.linear2(x))
    return self.dropout(self.linear3(x))
  

def train(nb_epochs=10, name_dataset='dyda_da', patience=2):
  train_loader, val_loader, test_loader = get_dataLoader(name_dataset=name_dataset)
  val_loss_list = []
  val_acc_list = []
  best_validation_acc = 0.0
  p = 0
  for epoch in range(1, nb_epochs + 1): 
      my_model.train()
      losses = []
      accuracies = []
      # use tqdm to better visualize the training process
      with tqdm(train_loader, unit="batch", position=0, leave=True) as tepoch: # evaluate every batch
          for data in tepoch:
              tepoch.set_description(f"Epoch {epoch}") # custome the printed message in tqdm
              optimizer.zero_grad()
              input = [ele[0] for ele in data["Utterance"]]
              input = tokenizer(input, return_tensors="pt", padding='max_length', max_length=30, truncation=True).to(device)
              label = torch.squeeze(data['Label']).to(device)
              output = my_model.forward(input)
              loss = criterion(output, label)
              loss.backward() # backward propagation
              torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1.0) # prevent exploding gradient 
              optimizer.step() # update parameters
              output = torch.argmax(F.softmax(output, dim=1), dim=1)
              losses.append(loss.item())
              accuracy = torch.sum(output == label).item() / label.size(0)
              accuracies.append(accuracy)
              # custome what is printed in tqdm message
              tepoch.set_postfix(loss=sum(losses)/len(losses), accuracy=100. * sum(accuracies)/len(accuracies))

      val_acc, val_loss = evaluate(val_loader)
      val_loss_list.append(val_loss)
      val_acc_list.append(val_acc)
      print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Val Accuracy: {:3.2f}%"
            .format(epoch, sum(losses)/len(losses), 100.*val_acc))
      test_acc, test_loss = evaluate(test_loader)
      print("Test loss: {:.4f}, Test Accuracy: {:3.2f}%"
                .format(test_loss, 100.*test_acc))
      if val_acc >= best_validation_acc:
          best_validation_acc = val_acc
          print("Validation accuracy improved, reset p = 0")
          p = 0
      else:
        p += 1
        if p == patience:
          print("Validation accuracy did not improve for {} epochs, stopping training...".format(patience))
          break
  print("Done")
  return val_loss_list, val_acc_list

def evaluate(data_loader):
  my_model.eval()
  losses = []
  correct = 0
  total = 0
  for data in data_loader:
    input = [ele[0] for ele in data["Utterance"]]
    input = tokenizer(input, return_tensors="pt", padding='max_length', max_length=30, truncation=True).to(device)
    label = torch.squeeze(data['Label']).to(device)
    output = my_model.forward(input)
    loss = criterion(output, label)
    output = torch.argmax(F.softmax(output, dim=1), dim=1)
    losses.append(loss.item())
    correct += torch.sum(output == label).item()
    total += len(label)
  return correct/total, sum(losses)/len(losses)


if __name__ == "__main__":
    import pickle
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # tasks = ["dyda_da", "sem", "maptask", "meld_e", "meld_s"]
    # nums_class = [4, 3, 12, 7, 3]
    task, num_class = 'sem', 3
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    # for num_class, task in zip(nums_class, tasks):
    model = DebertaModel.from_pretrained("microsoft/deberta-base")
    my_model = MyModel(trained_transformer=model, num_class=num_class, fine_tune_whole=True).to(device)
    my_model = my_model.double()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(), lr=2e-5, eps=1e-6)
    val_loss_list, val_acc_list = train(nb_epochs=10, name_dataset=task)
    with open(f'DeBERTa_LSTM_{task}_val_loss.pickle', 'wb') as handle:
      pickle.dump(val_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'DeBERTa_LSTM_{task}_val_acc.pickle', 'wb') as handle:
      pickle.dump(val_acc_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

