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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embed_size)
        self.lin_out = nn.Linear(hidden_size, vocab_size)
                            
    def forward(self, features, captions):
        captions_embeded = self.embed(captions[:, :-1])
        x = torch.cat([features.unsqueeze(1), captions_embeded], dim=1)
        outputs, _ = self.lstm(x)  # N, L, Hid
        logits = self.lin_out(outputs) # N, L, Voc
        return logits

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Assumes inputs is shape N, L, C
        predicts = []
        for i in range(max_len):
            if i == 0:
                outputs, (h, c) = self.lstm(inputs)  # Keep the states, use 0 default at first step
            else:
                outputs, (h, c) = self.lstm(inputs, (h, c))  # Pass the states to the LSTM
            # Predict
            logits = self.lin_out(outputs)
            new_word = logits.argmax(2).long()
            predicts.append(new_word.item())
            # Update new inputs
            inputs = self.embed(new_word)
            
        return predicts