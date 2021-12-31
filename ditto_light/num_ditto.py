import os
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

from .dataset_num import NumDittoDataset
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup, BertConfig
from tensorboardX import SummaryWriter
# from apex import amp

from transformers import BertForSequenceClassification
from ditto_light.classification_NN import classification_NN
from torch.nn import CosineSimilarity, BCEWithLogitsLoss, Sigmoid


class NumDittoCrossencoder(BertForSequenceClassification):
    """
    reference BertForTokenClassification class in the hugging face library
    https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertForSequenceClassification
    """
    def __init__(self, config, alpha_aug=0.8):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        if (config.num_input_dimension != 1):
            cos = CosineSimilarity()
            self.calculate_similiarity = lambda a, b: cos(a,b).view(-1,1)
            config.num_input_dimension = 1
        else:
            self.calculate_similiarity = self.calculate_difference
            
        self.classifier = classification_NN(
            inputs_dimension = config.num_input_dimension + config.text_input_dimension,
            num_hidden_lyr = config.num_hidden_lyr,
            dropout_prob = 0.2,
            bn =True
            )
        
        self.init_weights()
        self.loss_fct  = BCEWithLogitsLoss()
        
    
    def forward(
            self,
            numerical_featuresA,
            numerical_featuresB,
            input_ids,
            attention_mask,
            labels,
            token_type_ids):
        
        # compute the cls embedding of the text features
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            )
        cls_output = self.dropout(output[1])
        
        # calculate cossine similiary of numeric features
        numerical_features = self.calculate_similiarity(
            numerical_featuresA,
            numerical_featuresB)
        
        # Combined the text embedding with the similarity factor of numeric features
        all_features = torch.cat((cls_output, numerical_features.view(-1,1)), dim=1)

        return self.classifier(all_features)
    
    def calculate_difference(self, tensorA, tensorB):
        #return torch.abs(tensorA - tensorB)
        return (tensorA - tensorB)

def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_y = []
    all_probs = []
    calculate_prediction = Sigmoid()
    with torch.no_grad():
        for batch in iterator:
            input_ids, labels, attention_mask, token_type_ids, num_1, num_2 = batch
            logits = model(numerical_featuresA = num_1,
                            numerical_featuresB = num_2,
                            input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels,
                            token_type_ids  = token_type_ids)
            probs = calculate_prediction(logits)
            all_probs += probs.cpu().numpy().tolist()
            all_y += labels.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            #new_f1 = metrics.f1_score(all_y, pred, zero_division=1, average="micro")
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp, device):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.BCEWithLogitsLoss()
    
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        input_ids, labels, attention_mask, token_type_ids, num_1, num_2 = batch
        prediction = model(numerical_featuresA = num_1,
                            numerical_featuresB = num_2,
                            input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels,
                            token_type_ids  = token_type_ids)

        loss = criterion(prediction, labels.float().view(-1,1).to(device))

        # if hp.fp16:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     loss.backward()
        loss.backward()
            
        optimizer.step()
        scheduler.step()
        if i % 10 == 0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp, num_hidden_lyr=2):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = setup_cuda()
    
    # Build config for the bert classificaiton model
    bert_config = build_bert_config(
        len(trainset.num_pairs[0][0]),
        hp.lm,
        num_hidden_lyr)
    
    model = NumDittoCrossencoder(config = bert_config,
                                 alpha_aug=hp.alpha_aug)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # if hp.fp16:
    #    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(1, hp.n_epochs+1):
        # train
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp, device)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()

def build_bert_config(num_input_dimension, 
                      lm, 
                      num_hidden_lyr,
                      attention_probs_dropout_prob = 0.2,
                      hidden_dropout_prob = 0.1):
    config = BertConfig.from_pretrained(lm, num_labels=2)
    config.text_input_dimension = config.hidden_size
    config.num_input_dimension = num_input_dimension
    config.num_hidden_lyr = num_hidden_lyr
    config.lm = lm
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    config.hidden_dropout_prob = hidden_dropout_prob
    return config

def setup_cuda():
  if torch.cuda.is_available():    
      print('Running on GPU')
      return torch.device("cuda") 
  else:
      print('Running on CPU')
      return torch.device("cpu")