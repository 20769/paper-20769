import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import torch
import os

class ModelVisualizer:
    def __init__(self, model=None, writer=None):
        self.colors = {
            'high': '#2563eb',  
            'low': '#9ca3af'    
        }
        self.model = model
        self.writer = writer
        self.epoch_i = 0
    
    def get_output(self, input_tensor, times, i=0):
        with torch.no_grad():
            model_output = self.model(input_tensor, times, interp=True)
        for key, value in model_output.items():
            if isinstance(value, torch.Tensor):
                model_output[key] = value.cpu().detach()
        self.model_output = model_output
        self.input_tensor = input_tensor.detach().cpu()

    def merge_encodings(self, encoding, mask):

        bs, seg, m, n = encoding.shape

        encoding_flat = encoding.view(bs * seg, m, n)
        mask_flat = mask.view(bs * seg)

        selected_encodings = encoding_flat[mask_flat]

        merged_encodings = selected_encodings.view(-1, m, n)

        return merged_encodings
    
    def visualize_encodings(self, high_attention_encoding, low_attention_encoding, mask):
        high_flat = self.merge_encodings(high_attention_encoding, mask).squeeze()
        low_flat = self.merge_encodings(low_attention_encoding, ~mask).squeeze() if low_attention_encoding is not None else None
        
        if low_flat is not None:
            combined = np.vstack([high_flat, low_flat])
            labels = ['high'] * len(high_flat) + ['low'] * len(low_flat)
        else:
            combined = high_flat
            labels = ['high'] * len(high_flat)

        tsne = TSNE(n_components=2, random_state=42)
        embedded = tsne.fit_transform(combined)
        

        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            mask = np.array(labels) == label
            plt.scatter(embedded[mask, 0], embedded[mask, 1], 
                        c=self.colors[label], label=label, alpha=0.6)
        
        plt.legend()
        plt.title('High vs Low Attention Encodings' if low_flat is not None else 'High Attention Encodings')

        if self.writer is not None:
            self.writer.add_figure('Encodings', plt.gcf(), global_step=self.epoch_i)
        
        return embedded, labels
    
    def find_dense_regions(self, embedded, labels, min_samples=2):

        dbscan = DBSCAN(eps=0.5, min_samples=min_samples)
        clusters = dbscan.fit_predict(embedded)
        
        dense_indices = []
        for cluster_id in range(max(clusters) + 1):
            cluster_mask = clusters == cluster_id
            if np.sum(cluster_mask) >= min_samples:
                dense_indices.extend(np.where(cluster_mask)[0])
                
        return dense_indices
    
    def del_zeros(self, sequences, embedded):
        valid_length = np.sum(np.any(sequences != 0, axis=2), axis=1)
        sequences = sequences[np.where(valid_length > 0)]
        embedded = embedded[np.where(valid_length > 0)]
        
        return sequences, embedded
    
    def visualize_sequences(self, sequences, attention_values=None, title='Sequences'):
        n_seqs = len(sequences)
        n_features = sequences[0].shape[-1]
        
        fig, axes = plt.subplots(n_seqs, 1, figsize=(15, n_seqs * 2))
        if n_seqs == 1:
            axes = [axes]
            
        for i, (seq, ax) in enumerate(zip(sequences, axes)):
            seq = seq.numpy()
            valid_length = np.sum(np.any(seq != 0, axis=1))
            seq = seq[:valid_length]
            
            for feature_idx in range(seq.shape[1]):
                ax.plot(seq[:, feature_idx], label=f'Feature {feature_idx+1}')

                if attention_values is not None and i < len(attention_values):
                    att = attention_values[i]
                    ax.set_title(f'Sequence {i+1} (Attention: {att:.3f})')
                else:
                    ax.set_title(f'Sequence {i+1}')
            ax.legend()
                
        plt.tight_layout()
        plt.suptitle(title)
        if self.writer is not None:
            self.writer.add_figure('Sequences', plt.gcf(), global_step=self.epoch_i)
                
        
    def visualize_segmented_sequence(self, sequence, segment_info, batch_id):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.colors import Normalize, LinearSegmentedColormap
        batch_segments = [s for s in segment_info if s['batch_id'] == batch_id]
        if not batch_segments:
            raise ValueError(f"No segments found for batch_id {batch_id}")

        t, n = sequence.shape
        colors = plt.cm.tab10.colors


        cmap_alpha = LinearSegmentedColormap.from_list(
            'alpha_blend',
            [(0.1, 0.4, 0.8, 0.1), (0.1, 0.4, 0.8, 0.8)]
        )

        attentions = [seg['attention_mean'] for seg in batch_segments]
        norm = Normalize(vmin=min(attentions), vmax=max(attentions))

        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        if n == 1:
            ax.plot(np.arange(t), sequence[:, 0], color='black', linewidth=1)
        else:
            for i in range(n):
                ax.plot(np.arange(t), sequence[:, i], 
                        color=colors[i % len(colors)],
                        linewidth=1,
                        label=f'Series {i+1}')

        for seg in batch_segments:
            alpha = norm(seg['attention_mean']) * 0.7 + 0.1 
            ax.axvspan(seg['start'], seg['end'],
                       color=(0.1, 0.4, 0.8), 
                       alpha=alpha,
                       label=f"Seg{seg['seg_id']}")

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_alpha)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
        ax.set_xlabel('Time Step', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Time Series with Segmented Attention', fontsize=14)
        if n > 1:
            ax.legend(loc='upper right', 
                     frameon=True,
                     framealpha=0.9,
                     edgecolor='black')
                
        plt.tight_layout()
        if self.writer is not None:
            self.writer.add_figure('Segmented Sequence', plt.gcf(), global_step=self.epoch_i)
        
    def visualize_all(self, model_output=None, input_tensor=None, epoch_i=None, sample_batch_id=0):
        self.epoch_i = epoch_i
        if model_output is None:
            model_output = self.model_output
        else:
            for key, value in model_output.items():
                if isinstance(value, torch.Tensor):
                    model_output[key] = value.cpu().detach()

        if input_tensor is None:
            input_tensor = self.input_tensor
        else:
            input_tensor = input_tensor.cpu()           
        embedded, labels = self.visualize_encodings(
            model_output['high_attention_encoding'],
            model_output['low_attention_encoding'],
            model_output['mask']
        )
        

        dense_indices = self.find_dense_regions(embedded, labels)
        high_attention_seq = self.merge_encodings(model_output['all_seq'], model_output['mask'])
        low_attention_seq = self.merge_encodings(model_output['all_seq'], ~model_output['mask'])
        if len(dense_indices) >= 10:
            dense_indices = dense_indices[:10] 
            sequences = []
            attention_values = []
            for idx in dense_indices:
                if idx < len(high_attention_seq):
                    sequences.append(high_attention_seq[idx])
                    attention_values.append(1.0)
                else:
                    adjusted_idx = idx - len(high_attention_seq)
                    sequences.append(low_attention_seq[adjusted_idx])
                    attention_values.append(0.3)
            
            self.visualize_sequences(sequences, attention_values, 
                                   title='Dense Region Sequences')
        if model_output['low_attention_encoding'] is not None:
            self.visualize_segmented_sequence(
                input_tensor[sample_batch_id],
                model_output['seg_info'],
                sample_batch_id
            )

def generate_mock_data(batch_size=32, sequence_length=20, concept_dim=10, subseq_len=5):
    input_tensor = torch.rand(batch_size, sequence_length, concept_dim)

    high_attention_encoding = torch.rand(batch_size, sequence_length, concept_dim)
    low_attention_encoding = torch.rand(batch_size, sequence_length, concept_dim)
    all_attention_encoding = high_attention_encoding + low_attention_encoding

    high_attention_seq = torch.rand(batch_size, sequence_length, subseq_len, 1)
    low_attention_seq = torch.rand(batch_size, sequence_length, subseq_len, 1)

    segment_info = []
    for b in range(batch_size):
        for seg_idx in range(sequence_length//5):
            start = seg_idx * 5
            end = start + 5
            attention_mean = np.random.rand()
            segment_info.append({
                'batch_id': b,
                'seg_id': seg_idx,
                'start': start,
                'end': end,
                'attention_mean': attention_mean
            })

    return {
        'input': input_tensor,
        'high_attention_encoding': high_attention_encoding,
        'low_attention_encoding': low_attention_encoding,
        'all_attention_encoding': all_attention_encoding,
        'high_attention_seq': high_attention_seq,
        'low_attention_seq': low_attention_seq,
        'segment_info': segment_info
    }

def ts_forcast_visualize(sec, pred, label=None, writer=None, epoch=0):
    random = np.random.randint(0, sec.shape[0])
    b, t, n = sec.shape
    pred_len = pred.shape[1]
    # plot n_features in one figure
    for i in range(n-1):
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(t), sec[random, :, i], label='True', color='#1f77b4', linestyle='-')  # Blue
        plt.plot(np.arange(t, t+pred_len), pred[random, :, i], label='Pred', color='#ff7f0e', linestyle='-')  # Orange
        if label is not None:
            plt.plot(np.arange(t, t+pred_len), label[random, :, i], label='Label', color='#1f77b4', linestyle='--')  # Green
        plt.title(f'Feature {i+1} Forecast')
        plt.legend()
        if writer is not None:
            writer.add_figure(f'Feature {i+1} Forecast', plt.gcf(), global_step=epoch)
        else:
            plt.show()
            save_path = os.path.join(os.getcwd(), 'forecast')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, f'feature_{i+1}_forecast.png'))
    return
