import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # translating the initalisation parameters to class parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # initialising the embedding layer    -> outputs a vectore of dimension [1, embed_size]
        self.embed_rnn = nn.Embedding(vocab_size, embed_size)
        
        # initialising the LSTM layer
        # This layer takes the input as embed_size and output a vectore of dimension [1, hidden_size]
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size=hidden_size, 
                            num_layers = num_layers, 
                            batch_first = True)  # adding the batch first parameter to indicate the batch dimension
        
        # final Linear layer to map the hidden output to the vocab_size output
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeds = self.embed_rnn(captions[:,:-1])
        
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)
        
        lstm_out, _ = self.lstm(inputs)
        
        output = self.fc(lstm_out)
        
        return output
        

    def sample(self, inputs, states=None, max_len=10):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        caption = []
        
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))

        
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) 
            outputs = self.fc(lstm_out)        
            outputs = outputs.squeeze(1)                 
            wordid  = outputs.argmax(dim=1)              
            caption.append(wordid.item())
            
            
            inputs = self.embed_rnn(wordid.unsqueeze(0))  
            
        return caption