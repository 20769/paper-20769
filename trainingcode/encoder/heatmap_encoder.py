import torch
from torch import nn
import sys
from os.path import dirname as opd
import torch.utils.checkpoint as cp
sys.path.append(opd(opd(__file__)))

from layers.pos_Trimmer import SequenceTrimmer
from layers.Data_embedding import DataEmbedding
from layers.TCN_Enc import TCNEncoder

class HeatmapEncoder(nn.Module):
    def __init__(self, args, graph=None, device='cpu'):
        super(HeatmapEncoder, self).__init__()
        self.input_dim = args.input_dim
        self.concept_dim = args.concept_dim
        self.num_classes = args.num_classes
        self.max_len = args.max_len
        self.d_model = args.d_model
        self.encoder_type = args.encoder_type
        self.task_type = args.task_type
        self.args = args
        self.device = device
        if args.task_type == 'forcast':
            self.graph = graph.cpu()
        
        self.sequence_trimmer = SequenceTrimmer(self, args)
        self.sequence_trimmer.init_refer_model(args)
        
        
        self.encoder = TCNEncoder(input_size=self.d_model, 
                                hidden_size=args.hidden_size, 
                                num_layers=args.n_layers, 
                                output_size=self.concept_dim)
        self.encoder = self.encoder
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(self.concept_dim, self.concept_dim)
        self.apply_causal_mask()
        self.to(self.device)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def apply_causal_mask(self):
        if self.task_type == 'classify':
            self.data_embeddings = DataEmbedding(self.input_dim, self.d_model, selected_channels=None)
        else:
            if hasattr(self, 'graph'):
                graph = self.graph.numpy() if torch.is_tensor(self.graph) else self.graph
                self.data_embeddings = nn.ModuleList([
                    DataEmbedding(self.input_dim, self.d_model, selected_channels=graph[i])
                    for i in range(graph.shape[0])
                ])
                
    def mask_causal(self, mask):
        graph = self.graph  # [n_dim, n_dim], with 0-1
        device = mask.device
        n_dim = graph.shape[0]

        graph = graph.to(device)
        mask_list = []

        for i in range(n_dim):
            graph_i = graph[i]  
            graph_i_expanded = graph_i.unsqueeze(0).unsqueeze(0)  
            mask_i = mask.clone()
            mask_i = mask_i * graph_i_expanded  
            mask_i[..., graph_i == 0] = False       
            mask_list.append(mask_i.unsqueeze(0))    
        mask_stacked = torch.cat(mask_list, dim=0)
        mask_final = mask_stacked.permute(1, 2, 0, 3)

        return mask
    
    def train_refer_model(self, train_loader, val_loader, writter, save_path=None):
        if self.sequence_trimmer.model != None:
            refer_model = self.sequence_trimmer.model
            n_classes = self.num_classes
            task_type = self.task_type
        refer_model.train_refer_model(train_loader, val_loader, n_classes, epochs=100, task_type=task_type, writter=writter, save_path=save_path)

    def _encode_sequences_forast(self, sequences, masked=True, epoch=None, total_epochs=None):
        output = []
        chunk_size = 4  
        for i in range(0, self.num_classes, chunk_size):
            chunk_end = min(i + chunk_size, self.num_classes)
            chunk_embeddings = []
            
            for j in range(i, chunk_end):
                embedding = self.data_embeddings[j](sequences)
                chunk_embeddings.append(embedding)
            
            embeddings_chunk = torch.stack(chunk_embeddings, dim=-1)
            bs, t1, t2, n, m = embeddings_chunk.size()

            for j in range(chunk_end - i):
                embedding = embeddings_chunk[:, :, :, :, j].reshape(bs * t1, t2, n)
                outputs = self.encoder(embedding)
                
                outputs = self.fc(outputs)
                outputs = outputs.reshape(bs, t1, self.concept_dim)
                output.append(outputs)

            del embeddings_chunk, chunk_embeddings
            torch.cuda.empty_cache()
        
        return torch.stack(output, dim=-1)

    def _encode_sequences_classify(self, sequences, masked=True, epoch=None, total_epochs=None):
        embedding = self.data_embeddings(sequences)
        bs, t1, t2, m = embedding.size()
        
        batch_size = 128 
        outputs = []
        
        for i in range(0, bs * t1, batch_size):
            end_idx = min(i + batch_size, bs * t1)
            batch_embedding = embedding.view(bs * t1, t2, m)[i:end_idx]
            
            batch_output = cp.checkpoint(self.encoder, batch_embedding, use_reentrant=False,)
            outputs.append(batch_output)
            
            del batch_embedding
            torch.cuda.empty_cache()
        
        output = torch.cat(outputs, dim=0)
        return output.view(bs, t1, -1)


    def get_regularization_loss(self):
        l1_loss = 0
        l2_loss = 0
        for name, param in self.named_parameters():
            if 'bias' not in name:
                l1_loss += param.abs().sum()
                l2_loss += (param ** 2).sum()
        return (l1_loss + l2_loss) / 100

    def forward(self, seq, times, epoch=None, total_epochs=None):
        with torch .no_grad():
            segment_info, all_seq, mask = self.sequence_trimmer(seq, times)
        if self.task_type == 'forcast':
            mask = self.mask_causal(mask)
            outputs = self._encode_sequences_forast(all_seq, masked=True, epoch=epoch, total_epochs=total_epochs)
        else:
            outputs = self._encode_sequences_classify(all_seq, masked=True, epoch=epoch, total_epochs=total_epochs)
        torch.cuda.empty_cache()
        return outputs, all_seq, mask, segment_info

    def _clip_gradients(self, outputs):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return outputs

