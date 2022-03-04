import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence


class GRU(nn.Module):

    def __init__(self, args, num_classes, input_size=1, hidden_size=512, num_layers=1):
        super(GRU, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        if args.bidirectional:
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.2)
            self.linear = nn.Linear(2*hidden_size, num_classes)
        else:
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_sequences, hidden_state_0):      
        gru_out, h_out = self.gru(input_sequences, hidden_state_0)
        gru_out_padded, gru_out_lengths = pad_packed_sequence(gru_out, batch_first=True)
        
        # # The logits we are getting are for a sequence of 6 visits. However, we only want the logits for last valid admission, so we only return that one
        logits = self.linear(gru_out_padded)
        # print(f"Shape of the FullyConnected output {logits.shape}") #-> [BatchSize, 6 (numVisits with padding), NumLabels]
        # print(f"Shape of the FullyConnected output {logits}")
        
        if self.num_classes == 1:
            corrected_logits =  torch.zeros(len(gru_out_lengths.tolist()), dtype=torch.float)
            for i, (logit, length) in enumerate(zip(logits, gru_out_lengths.tolist())):
                # print(logit[length-1], logit[length-1][0])
                corrected_logits[i] = logit[length-1][0]
                
        elif self.num_classes > 1:
            corrected_logits =  torch.zeros((len(gru_out_lengths.tolist()), self.num_classes), dtype=torch.float)
            for i, (logit, length) in enumerate(zip(logits, gru_out_lengths.tolist())):
                corrected_logits[i] = logit[length-1]

        return corrected_logits
    

class LSTM(nn.Module):

    def __init__(self, args, num_classes, input_size=1, hidden_size=512, num_layers=1):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        if args.bidirectional:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.2)
            self.linear = nn.Linear(2*hidden_size, num_classes)
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, input_sequences, hidden_state_0, cell_state_0):      
        lstm_out, _ = self.lstm(input_sequences, (hidden_state_0, cell_state_0))
        lstm_out_padded, lstm_out_lengths = pad_packed_sequence(lstm_out, batch_first=True)
        
        logits = self.linear(lstm_out_padded)
        
        if self.num_classes == 1:
            corrected_logits =  torch.zeros(len(lstm_out_lengths.tolist()), dtype=torch.float)
            for i, (logit, length) in enumerate(zip(logits, lstm_out_lengths.tolist())):
                # print(logit[length-1], logit[length-1][0])
                corrected_logits[i] = logit[length-1][0]
                
        elif self.num_classes > 1:
            corrected_logits =  torch.zeros((len(lstm_out_lengths.tolist()), self.num_classes), dtype=torch.float)
            for i, (logit, length) in enumerate(zip(logits, lstm_out_lengths.tolist())):
                corrected_logits[i] = logit[length-1]

        return corrected_logits
        