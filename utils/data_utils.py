import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords

class DocumentSentimentDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {'positive': 1, 'negative': 0}
    INDEX2LABEL = {1: 'positive', 0: 'negative'}
    NUM_LABELS = 2
    
    # def load_dataset(self, path): 
    #     df = pd.read_csv(path, sep='\t', header=None)
    #     df.columns = ['text','label']
    #     df['label'] = df['label'].apply(lambda lab: self.LABEL2INDEX[lab])
    #     return df
    
    def __init__(self, df, tokenizer, no_special_token=False, max_seq_len=512, *args, **kwargs):
        self.data = df
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token
        self.max_seq_len = max_seq_len
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        text, label = data['text'], data['label']
        
        stop_words = set(stopwords.words('english'))

        text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
        subwords = self.tokenizer.encode(text, 
                                        add_special_tokens=not self.no_special_token,         
                                        max_length=self.max_seq_len, # Enforce max_seq_len here
                                        truncation=True, # Truncate if exceeding max_seq_len
                                        padding='max_length')# Pad to max_seq_len

        return np.array(subwords), np.array(label), data['text']
    
    def __len__(self):
        return len(self.data)

class DocumentSentimentDataLoader(DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(DocumentSentimentDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len
        
    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        
        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        sentiment_batch = np.zeros((batch_size, 1), dtype=np.int64)
        
        seq_list = []
        for i, (subwords, sentiment, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_batch[i,:len(subwords)] = subwords
            mask_batch[i,:len(subwords)] = 1
            sentiment_batch[i,0] = sentiment
            
            seq_list.append(raw_seq)
            
        return subword_batch, mask_batch, sentiment_batch, seq_list   