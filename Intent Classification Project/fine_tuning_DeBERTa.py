from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaModel, AutoTokenizer
from datasets import load_dataset

def get_dataLoader(name_dataset='dyda_da', batch_size=64):
  dataset = load_dataset('silicone', name_dataset, split='train')
  dataset_val = load_dataset('silicone', name_dataset, split='validation')
  dataset_test = load_dataset('silicone', name_dataset, split='test')

  training_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
  val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)
  test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2, shuffle=True, drop_last=True)

  return training_loader, val_loader, test_loader

class MyModel(torch.nn.Module):
  def __init__(self, trained_transformer, num_class, drop_out=0.15, fine_tune_whole=False):
    super().__init__()
    self.trained_transformer = trained_transformer
    self.linear1 = torch.nn.Linear(768, 768)
    self.linear2 = torch.nn.Linear(768, 768)
    self.linear3 = torch.nn.Linear(768, num_class)
    self.dropout = torch.nn.Dropout(p=drop_out)
    self.relu = torch.nn.ReLU()

    # Fix the pre-trained model
    if not fine_tune_whole:
      for param in self.trained_transformer.parameters():
        param.requires_grad = False
  
  def forward(self, inputs):
    x = self.trained_transformer(**inputs).last_hidden_state[:, 0] # Taking the  vector of first token <CLS> as representation of whole sentence
    x = self.relu(self.linear1(x))
    x = self.relu(self.linear2(x))
    return self.dropout(self.linear3(x))
  

def train(nb_epochs=10, name_dataset='dyda_da', batch_size=32, patience=2):

  train_loader, val_loader, test_loader = get_dataLoader(name_dataset=name_dataset, batch_size=batch_size)
  best_validation_acc = 0.0
  val_acc_list = []
  val_loss_list = []
  for epoch in range(1, nb_epochs + 1): 
      my_model.train()
      losses = []
      accuracies = []
      # use tqdm to better visualize the training process
      with tqdm(train_loader, unit="batch", position=0, leave=True) as tepoch: # evaluate every batch
          for data in tepoch:
              tepoch.set_description(f"Epoch {epoch}") # custome the printed message in tqdm
              optimizer.zero_grad()
              input = tokenizer(data['Utterance'], return_tensors="pt", padding='max_length', max_length=30, truncation=True).to(device)
              label = data['Label'].to(device)
              output = my_model.forward(input)
              loss = criterion(output, label)
              loss.backward() # backward propagation
              torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1.0) # prevent exploding gradient 
              optimizer.step() # update parameters
              
              output = torch.argmax(F.softmax(output, dim=1), dim=1)
              losses.append(loss.item())
              accuracy = torch.sum(output == label).item() / len(label)
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
		  # save the model or not
      if val_acc >= best_validation_acc:
          best_validation_acc = val_acc
          print("Validation accuracy improved, reset p = 0")
          # torch.save({
          #   'model_state_dict': my_model.state_dict(),
          #   'optimizer_state_dict': optimizer.state_dict(),
          #   }, f"./best_deberta_{name_dataset}.pt")
          p = 0
      else:
        p += 1
        if p == patience:
          print("Validation accuracy did not improve for {} epochs, stopping training...".format(patience))
          break

  # torch.save({
  #     'model_state_dict': my_model.state_dict(),
  #     'optimizer_state_dict': optimizer.state_dict(),
  #     }, f"./last_deberta_{name_dataset}.pt") 
  # print("Loading best checkpoint...")    
  # my_model.load_state_dict(torch.load(f"./best_deberta_{name_dataset}.pt")['model_state_dict'])
  print("Done")
  return val_loss_list, val_acc_list

def evaluate(data_loader):
  
  my_model.eval()
  losses = []
  correct = 0
  total = 0
  for data in data_loader:
    input = tokenizer(data['Utterance'], return_tensors="pt", padding='max_length', max_length=30, truncation=True).to(device)
    label = data['Label'].to(device)
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
    print(f"Current device is {device}")
    tasks = ["dyda_da", "sem", "maptask", "meld_e", "meld_s"]
    nums_class = [4, 3, 12, 7, 3]
    # num_class, task = 4, "dyda_da"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    for num_class, task in zip(nums_class, tasks):
      model = DebertaModel.from_pretrained("microsoft/deberta-base")
      my_model = MyModel(trained_transformer=model, num_class=num_class, fine_tune_whole=True).to(device)
      my_model = my_model.double()
      criterion = torch.nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(my_model.parameters(), lr=2e-5, eps=1e-6)
      val_loss_list, val_acc_list = train(nb_epochs=10, name_dataset=task, batch_size=64)
      with open(f'DeBERTa_{task}_val_loss.pickle', 'wb') as handle:
        pickle.dump(val_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(f'DeBERTa_{task}_val_acc.pickle', 'wb') as handle:
        pickle.dump(val_acc_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

