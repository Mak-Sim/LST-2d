import torch
import torch.nn as nn


class L2DST(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.dense2 = nn.LazyLinear(ffn_num_outputs)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)

    def forward(self, X):
        out = torch.nn.functional.tanh(self.drop1(self.dense1(X)))
        out1 = torch.transpose(out, -1, -2)
        out = torch.nn.functional.tanh(self.drop2(self.dense2(out1)))
        out = torch.transpose(out, -2, -1)
        return out

    def get_embeddings(self,X):
        out = torch.nn.functional.tanh(self.dense1(X))
        e1 = out
        out = torch.transpose(e1, -1, -2)
        out = torch.nn.functional.tanh(self.dense2(out))
        e2 = out
        out = torch.transpose(e2, -2, -1)
        
        return e1,e2
    
class LST_1(nn.Module):
    def __init__(self, input_size, num_classes, device='cpu'):
        super(LST_1, self).__init__()
        
        self.num_classes = num_classes
        self.L2DST = L2DST(input_size,input_size)
        
        self.dropout = nn.Dropout(p=0.1)
        self.W_o = nn.Linear(input_size*input_size, num_classes, device=device)
        

        self.log_softmax = nn.LogSoftmax(dim=1)
        
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, x):
        # Multiply input by weights and add biases
        
        out = self.L2DST(x)
        
        out = out.reshape(out.shape[0],-1)
        
        out = self.log_softmax(self.dropout(self.W_o(out)))
        
        return out
    
    def get_embeddings(self,x):
        e1,e2 = self.L2DST.get_embeddings(x)
        
        return e1,e2
    

class LST_2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device='cpu'):
        super(LST_2, self).__init__()
        
        self.L2DST_1 = L2DST(input_size,hidden_size)
        self.L2DST_2 = L2DST(hidden_size,hidden_size)
        
        self.dropout = nn.Dropout(p=0.1)
        
        self.W_o = nn.Linear(hidden_size*hidden_size, num_classes, device=device)
        
        self.num_classes = num_classes

        self.log_softmax = nn.LogSoftmax(dim=1)
        
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, x):
        
        out = self.L2DST_2(self.L2DST_1(x))
        
        out = out.reshape(out.shape[0],-1)
        
        # Used with NLLLoss
        out = self.log_softmax(self.dropout(self.W_o(out)))
        
        return out
    
    
class ResLST(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device='cpu'):
        super(ResLST, self).__init__()
        
        self.L2DST_1 = L2DST(input_size,input_size)
        self.L2DST_2 = L2DST(input_size,hidden_size)
        self.L2DST_3 = L2DST(hidden_size,hidden_size)
        
        dropout = 0.1
        self.dropout = nn.Dropout(dropout)
        
        self.W_o = nn.Linear(hidden_size*hidden_size, num_classes, device=device)
        
        self.num_classes = num_classes

        self.log_softmax = nn.LogSoftmax(dim=1)
        
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, x):
        out1 = self.L2DST_1(x)
        out = self.L2DST_2(out1) 
        out = self.L2DST_3(out) + out1 
        
        out = out.reshape(out.shape[0],-1)
        
        out = self.log_softmax(self.dropout(self.W_o(out)))
        
        return out