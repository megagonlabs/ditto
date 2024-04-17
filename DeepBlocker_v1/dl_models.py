#This file contains a bunch of DL network definitions that are used by the tuple embedding models

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from configurations import * 
from tqdm import tqdm

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

#This is a simple dataset for loading numpy matrices
class NumPy_Dataset(Dataset):
    def __init__(self, embedding_matrix):
        self.embedding_matrix = embedding_matrix

    def __getitem__(self, index):
        return torch.tensor(self.embedding_matrix[index, :]).float()

    def __len__(self):
        return len(self.embedding_matrix)

#This is used for CTT and Hybrid where you want to load L_e, R_e and y_e
# where L_e and R_e are the numpy matrices for tuple pairs 
# that are either a perturbation of each other as denoted by y_e
class NumPy_Triplet_Dataset(Dataset):
    def __init__(self, left_embedding_matrix, right_embedding_matrix, labels):
        self.left_embedding_matrix = left_embedding_matrix
        self.right_embedding_matrix = right_embedding_matrix
        self.labels = labels
        if (len(left_embedding_matrix) != len(right_embedding_matrix)) \
            or (len(right_embedding_matrix) != len(labels)):
            raise Exception("The dimensions of left and right embedding matrix and labels do not match")

    def __getitem__(self, index):
        left = torch.tensor(self.left_embedding_matrix[index, :]).float()
        right = torch.tensor(self.right_embedding_matrix[index, :]).float()
        label = torch.tensor(self.labels[index]).float()
        return left, right, label

    def __len__(self):
        return len(self.left_embedding_matrix)

class AutoEncoder(nn.Module):
    #This model is assumed to be layered
    def __init__(self, input_dimension, hidden_dimensions):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1])
            )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dimensions[1], hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dimensions[0], input_dimension)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
 
    def get_tuple_embedding(self, t1):
        with torch.no_grad():
            #added by sr
            t1 = t1.to(get_device())
            return self.encoder(t1).detach().cpu().numpy()


class AutoEncoderTrainer:
    def __init__(self, input_dimension, hidden_dimensions):
        super(AutoEncoderTrainer, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def train(self, embedding_matrix, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model = AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(embedding_matrix)

        train_dataloader = DataLoader(dataset=NumPy_Dataset(embedding_matrix), batch_size=batch_size, shuffle=True)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)

        self.model.train()

        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, data in tqdm(enumerate(train_dataloader),leave=False):
                data = data.to(self.device)
                #added by sr
                #self.model = self.model.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                #print('output',output,'data',data)
                loss = loss_function(output, data)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_tuples))

        self.model.eval()

        self.save_model('./checkpoints/ae.pt')
        
        return self.model
    
    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model = AutoEncoder(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()
        
class CTTModel(nn.Module):
    #This model is assumed to be layered
    def __init__(self, input_dimension, hidden_dimensions):
        super(CTTModel, self).__init__()
        #print('input dims',input_dimension,'hidden dims',hidden_dimensions)
        self.siamese_summarizer = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimensions[0]),
            nn.ReLU(True),
            nn.Linear(hidden_dimensions[0], hidden_dimensions[1]),
            nn.ReLU(True)
            )
        #Simple Binary classifier 
        self.classifier = nn.Linear(hidden_dimensions[1], 1)


    def forward(self, t1, t2):
        #print('t1.shape',t1.shape)
        t1 = self.siamese_summarizer(t1)
        #print('t2.shape',t2.shape)
        t2 = self.siamese_summarizer(t2)
        pred = self.classifier(torch.abs(t1 - t2))
        return torch.sigmoid(pred)

    def get_tuple_embedding(self, t1):
        with torch.no_grad():
            return self.siamese_summarizer(t1).detach().numpy()

class CTTModelTrainer:
    def __init__(self, input_dimension, hidden_dimensions):
        super(CTTModelTrainer, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimensions = hidden_dimensions

    def train(self, left_embedding_matrix, right_embedding_matrix, labels, 
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
        self.model = CTTModel(self.input_dimension, self.hidden_dimensions)
        self.device = get_device()
        self.model.to(self.device)
        num_tuples = len(left_embedding_matrix)

        dataset = NumPy_Triplet_Dataset(left_embedding_matrix, right_embedding_matrix, labels)
        train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)

        self.model.train()

        for epoch in range(num_epochs):
            train_loss = 0
            for batch_idx, (left, right, label) in enumerate(train_dataloader):
                left = left.to(self.device)
                right = right.to(self.device)
                label = label.unsqueeze(-1)
                label = label.to(self.device)
                optimizer.zero_grad()
                #print('left.shape',left.shape,'right.shape',right.shape)
                output = self.model(left, right)
                loss = loss_function(output, label)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_tuples))
        
        self.model.eval()

        self.save_model('./checkpoints/ctt.pt')

        return self.model
    
    def save_model(self, output_file_name):
        torch.save(self.model.state_dict(), output_file_name)

    def load_model(self, input_file_name):
        self.model = CTTModel(self.input_dimension, self.hidden_dimensions)
        self.model.load_state_dict(torch.load(input_file_name))
        self.model.eval()
