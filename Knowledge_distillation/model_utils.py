

from utils import *
from tqdm import tqdm

from statistics import mean





class MyLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, return_sequences=False):
        super(MyLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_features = n_features
        self.return_sequences = return_sequences

        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(n_features, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, n_classes)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    def forward(self, y):
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        
        for i, time_step in enumerate(y.split(1, dim=1)):
            h_t, c_t = self.lstm1(torch.squeeze(time_step), (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            h_t2 = self.tanh(h_t2)
            output = self.linear(h_t2) # output from the last FC layer
            output = self.relu(output)
            outputs.append(output)
        
        if self.return_sequences:
            outputs = torch.stack(outputs, 1).squeeze(2)
        else:
            outputs = outputs[-1]

        return outputs


class HARNet(nn.Module): 

    def __init__(self, n_lstm_layers, n_features, n_hidden, n_classes):
        super(HARNet, self).__init__()
        self.n_lstm_layers = n_lstm_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_features = n_features

        # lstm1, lstm2, linear are all layers in the network
        self.lstms = []
        for i in range(n_lstm_layers):
            if i == 0:
                self.lstms.append(MyLSTM(n_features = n_features, n_hidden = self.n_hidden, n_classes = self.n_hidden, return_sequences = True))
            else:
                self.lstms.append(MyLSTM(self.n_hidden, self.n_hidden, self.n_hidden, return_sequences = False))
        self.lstms = nn.ModuleList(self.lstms)
        self.linear = nn.Linear(self.n_hidden, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x) : 
        for lstm in self.lstms:
            x = lstm(x)

        x = self.relu(x) 
        y = self.linear(x)
        y = self.softmax(y) 

        return y
    



def train(model, criterion, optimizer, train_loader, privacy_engine, device):
    accs = []
    losses = []
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        preds = logits.argmax(-1)
        n_correct = float(preds.eq(y.argmax(-1)).sum())
        batch_accuracy = n_correct / len(y)

        accs.append(batch_accuracy)
        losses.append(float(loss))

    if privacy_engine is not None:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()  # Yep, we put a pointer to privacy_engine into your optimizer :)
        print(f"(ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}")
    print(
        f"Train Accuracy: {mean(accs):.6f}"
        f"Train Loss: {mean(losses):.6f}"
    ) 
    return  


def test(model, test_loader, privacy_engine, device):
  accs = []
  with torch.no_grad():
    for x, y in tqdm(test_loader):
      x = x.to(device)
      y = y.to(device)

      preds = model(x).argmax(-1)
      n_correct = float(preds.eq(y.argmax(-1)).sum())
      batch_accuracy = n_correct / len(y)

      accs.append(batch_accuracy)

  print(f"Test Accuracy: {mean(accs):.6f}")
  if privacy_engine is not None : 
    epsilon, best_alpha = privacy_engine.get_privacy_spent()
    print(f"(ε = {epsilon:.2f}, δ = {privacy_engine.target_delta}) for α = {best_alpha}")
      
  
  return
