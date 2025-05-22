import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
from sklearn.metrics import roc_auc_score
args={
    'hidden_size': 128,
    'num_layers': 4,
}
class ReferModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 task_type, num_classes=None, pred_step=None):
        super(ReferModel, self).__init__()
        hidden_size = args['hidden_size']
        num_layers = args['num_layers']
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.proj_h = nn.Linear(hidden_size, hidden_size)  
        self.proj_x = nn.Linear(input_size, hidden_size)   

        self.attn_linear = nn.Linear(hidden_size, 1, bias=False)

        self.fc = nn.Linear(hidden_size, num_classes if task_type == 'classify' else input_size)

        self.dropout = nn.Dropout(0.3)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.task_type = task_type
        self.num_classes = num_classes
        self.pred_step = pred_step

        self.instancenorm_h = nn.InstanceNorm1d(hidden_size)
        self.instancenorm_x = nn.InstanceNorm1d(hidden_size)

    def forward(self, x, times=None):
        """
        x: [batch_size, T, input_size]
        """
        x, means, stdev = self.normalize(x)
        
        batch_size, seq_len, input_dim = x.size()

        x = self.instancenorm_x(x.transpose(1, 2)).transpose(1, 2)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        self.to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))  
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.instancenorm_h(lstm_out.transpose(1, 2)).transpose(1, 2)
        proj_h = self.proj_h(lstm_out)  
        proj_x = self.proj_x(x)
        proj_x = self.instancenorm_x(proj_x.transpose(1, 2)).transpose(1, 2)
        lstm_out_ex = proj_h.unsqueeze(2)
        x_ex = proj_x.unsqueeze(2).expand(-1, -1, input_dim, -1)
        lstm_out_ex = lstm_out_ex.expand(-1, -1, input_dim, -1)
        combined = torch.tanh(lstm_out_ex + x_ex)
        logits = self.attn_linear(combined)
        logits = logits.squeeze(-1)
        logits_2d = logits.view(batch_size, -1)  # [batch_size, T*N]
        alpha_2d = F.softmax(logits_2d, dim=1)   # [batch_size, T*N]
        attention_map = alpha_2d.view(batch_size, seq_len, input_dim)  # 最终的 2D 注意力分布
        if self.task_type == 'classify':
            context_vector = (lstm_out_ex * attention_map.unsqueeze(-1)).sum(dim=(1, 2))
            out = self.fc(context_vector)
        else:
            context_vector = (lstm_out_ex * attention_map.unsqueeze(-1)).sum(dim=(2))
            out = self.fc(context_vector)
            out = self.denormalize(out, means, stdev, seq_len)
        return out, attention_map

    def normalize(self, x):
        # Channel-wise Z-Score normalization
        means = x.mean(dim=(0, 1), keepdim=True).detach()
        stdev = x.std(dim=(0, 1), keepdim=True).detach() + 1e-5
        x = (x - means) / stdev
        return x, means, stdev

    def denormalize(self, x, means, stdev, length):
        if x.dim() == 2: 
            x = x * stdev[0, 0, :] + means[0, 0, :]
        elif x.dim() == 3: 
            x = x * stdev[0, 0, :].unsqueeze(0).unsqueeze(1).repeat(x.size(0), length, 1)
            x = x + means[0, 0, :].unsqueeze(0).unsqueeze(1).repeat(x.size(0), length, 1)
        return x

    def train_refer_model(self, train_loader, val_loader, n_classes, epochs=50, task_type='classify', writter=None, reload=True, save_path=None):
        self.to('cuda')

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        if not os.path.exists(save_path + '/refer.pt'):
            reload = False
        if reload:
            self.load_state_dict(torch.load(save_path + '/refer.pt'))
        else:
            if task_type == 'classify':
                class_count = torch.zeros(n_classes).to('cuda')
                for X, times, y in train_loader:
                    X, times, y = X.to('cuda'), times.to('cuda'), y.to('cuda')
                    class_count += torch.bincount(y, minlength=n_classes)
                class_weight = 1. / class_count
                loss_fn = nn.CrossEntropyLoss(weight=class_weight)
            else: 
                loss_fn = nn.MSELoss()
            
            optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-4)
            self.to('cuda')
            epochs = 300

            epoch_bar = tqdm.tqdm(range(len(train_loader)*epochs))
            batch_size = len(train_loader)
            best_val_loss = 1e10
            val_loss = 1e10
            for epoch in range(epochs):
                for X, times, y in train_loader:
                    X = X.squeeze(1)
                    times = times.squeeze(1)
                    X, times, y = X.to('cuda'), times.to('cuda'), y.to('cuda')
                    if task_type == 'regression':
                        y = y.float()
                    out, _ = self.forward(X)
                    optimizer.zero_grad()
                    loss = loss_fn(out, y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
                    
                    optimizer.step()
                    epoch_bar.set_description(f'Loss: {loss.item()}')
                    epoch_bar.update(1)
                    if epoch_bar.n % 500 == 0:
                        writter.add_scalar('refer/loss', loss.item(), epoch_bar.n)
                        with torch.no_grad():
                            self.eval()
                            val_loss = validate(self, epoch_bar.n, val_loader, task_type, writter, 'cuda')
                            self.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.state_dict(), save_path + '/refer.pt')
        return self
    
def validate(model, epoch, val_loader, task_type, writter, device):
    model.to(device)
    model.eval()
    val_loss = []
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss() if task_type == 'classify' else nn.MSELoss()
    out_list = []
    y_list = []
    with torch.no_grad():
        for X, times, y in val_loader:
            X = X.squeeze(1)
            times = times.squeeze(1)
            X, times, y = X.to(device), times.to(device), y.to(device)
            if task_type == 'regression':
                y = y.float()
            out, _ = model.forward(X)
            loss = loss_fn(out, y)
            val_loss.append(loss.item())
            out_list.append(out)
            y_list.append(y)
            if task_type == 'classify':
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
    val_loss = sum(val_loss) / len(val_loader)
    
    if task_type == 'classify':
        accuracy = 100 * correct / total
        y = torch.cat(y_list, dim=0)
        out = torch.cat(out_list, dim=0)
        # Convert outputs to binary format for roc_auc_score
        y_one_hot = F.one_hot(y, num_classes=out.size(1)).cpu().numpy()
        out_probs = F.softmax(out, dim=1).cpu().numpy()
        AUC = roc_auc_score(y_one_hot, out_probs, multi_class='ovr')
        writter.add_scalar('refer/val_loss', val_loss, epoch)
        writter.add_scalar('refer/val_accuracy', accuracy, epoch)
        writter.add_scalar('refer/val_AUC', AUC, epoch)
    else:
        y = torch.cat(y_list, dim=0)
        out = torch.cat(out_list, dim=0)
        mse = F.mse_loss(out, y).item()
        mae = F.l1_loss(out, y).item()
        writter.add_scalar('refer/val_loss', val_loss, epoch)
        writter.add_scalar('refer/val_mse', mse, epoch)
        writter.add_scalar('refer/val_mae', mae, epoch)

    model.train()
    return val_loss

def main():
    input_size = 10
    hidden_size = 20
    num_layers = 2
    task_type = 'classification'
    num_classes = 5
    pred_step = 3

    model = ReferModel(input_size, hidden_size, num_layers, task_type, num_classes, pred_step)
    model.to('cuda')

    # Create dummy data
    batch_size = 4
    seq_length = 15
    x = torch.randn(batch_size, seq_length, input_size).to('cuda')
    times = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).to('cuda')

    # Forward pass
    output, attention_weights = model.forward(x, times)
    print("Output:", output)
    print("Attention Weights:", attention_weights)

    # Grad-CAM
    heatmap = model.grad_cam(x)
    print("Heatmap:", heatmap)

if __name__ == "__main__":
    main()
