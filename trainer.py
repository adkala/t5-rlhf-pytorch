from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import random
import reward
import copy

class BaseTrainer:
    loss_history = []
    acc_history = []
    cur_epoch = 0

    def __init__(self, 
                 model,
                 optimizer,
                 dataloader,
                 interval=1):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.interval = interval

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.model.to(self.device)

    def train(self, n, verbose=True, manual=False):
        pbar = tqdm(range(n))
        for i in range(n):
            if not manual:
                loss = self._train_single_epoch(verbose)
            else:
                loss = self._train_single_epoch_manual()
            pbar.update(1)
            pbar.set_description('loss: %.8f' % loss)

            if i % self.interval == 0:
                self.loss_history.append(loss)
                self.acc_history.append(self.test(2)) # change test input
                if verbose:
                    print('acc: %.4f' % self.acc_history[-1])
                

    def test(self, n=10, max_length=512, verbose=True):
        total = 0
        for input, label in random.choices(self.dataloader.dataset, k=n):
            input_ids = self.model.tokenizer(input, return_tensors='pt').input_ids.to(self.device)
            prediction = self.model.tokenizer.decode(
                self.model.generate(input_ids, max_length=max_length)[0], 
                skip_special_tokens=True)
            acc = reward.base_reward(label, prediction)
            if verbose:
                print('acc: %.4f,\n\nlabel: %s\n\nprediction: %s' % (acc, label, prediction))
            total += acc
        return total / n

    def _train_single_epoch(self, verbose=True):
        for data in self.dataloader:
            inputs, labels = data
            t_inputs = self.model.tokenizer(inputs, padding=True, return_tensors='pt').to(self.device)
            t_labels = self.model.tokenizer(labels, padding=True, return_tensors='pt').to(self.device).input_ids
            
            t_labels[labels == self.model.tokenizer.pad_token_id] = -100
            
            outputs = self.model(input_ids=t_inputs.input_ids, attention_mask=t_inputs.attention_mask, labels=t_labels)
            self.optimizer.zero_grad()
            if isinstance(outputs, dict):
                loss = outputs.loss
            else:
                loss = outputs[1]
            loss.backward()
            self.optimizer.step()

            if verbose:
                print(loss.item())
        return loss.item()

    def _train_single_epoch_manual(self):
        for data in self.dataloader:
            inputs, labels = batch
            tokenized_inputs = model.tokenizer(inputs, padding=True, return_tensors='pt').to(torch.device('mps'))
            scores, out = model.generate(tokenized_inputs.input_ids, tokenized_inputs.attention_mask, max_length=4)
        
            tokenized_labels = model.tokenizer(inputs, padding=True, return_tensors='pt').to(torch.device('mps'))
            output = loss(scores.flatten(0, 1), tokenized_labels.input_ids[:, :scores.shape[1]].flatten(0, 1))

            self.optimizer.zero_grad()
            output.backward()
            self.optimizer.step()
        return output.item()


class PPOIter:
    def __init__(self, state, action, scores, ref_scores, prediction, label):
        self.state = state # token
        self.action = action # token
        self.scores = scores # dist
        self.ref_scores = ref_scores # dist
        self.prediction = prediction # dumped json
        self.label = label # dumped json

class ClipObjectiveLoss(torch.nn.Module):
    def __init__(self, clip_param=0.8, reward = reward.base_reward):
        super(ClipObjectiveLoss, self).__init__()
        self.clip_param = clip_param
        self.reward = reward
        
    def forward(self, ppoIter: PPOIter):
        return -1 * self._clip(ppoIter)

    def _clip(self, ppo_iter: PPOIter):
        advantage = self._advantage(ppo_iter.prediction, ppo_iter.label)
        ratio = self._prob_ratio(ppo_iter.scores, ppo_iter.ref_scores, ppo_iter.action)
        return min([ratio * advantage, torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage])
    
    def _advantage(self, prediction, label):
        t = 0
        for i in range(len(label)):
            t += self.reward(label[i], prediction[i])
        return t / len(label)
    
    def _prob_ratio(self, scores, ref_scores, action): # mean ratio per val in batch (action seq includes starting pad)
        logprobs, ref_logprobs = self._scores_to_prob(scores, ref_scores)
        
        log_action_probs = logprobs.gather(-1, action[:, 1:logprobs.shape[1]].unsqueeze(-1))
        ref_log_action_probs = ref_logprobs.gather(-1, action[:, 1:logprobs.shape[1]].unsqueeze(-1)) 

        return (log_action_probs - ref_log_action_probs).exp().sum() / logprobs.shape[0]
    
    def _kl_divergence(self, scores, ref_scores):
        logprobs, ref_logprobs = self._scores_to_prob(scores, ref_scores)

        logprobs = torch.flatten(torch.transpose(logprobs, 0, 1), 1, 2)
        ref_logprobs = torch.flatten(torch.transpose(ref_logprobs, 0, 1), 1, 2)
        
        mask = torch.zeros(*logprobs.shape).to(device) # masks out pad values if scores < ref_scores
        mask[:, :min_shape_0 * token_num] = 1  
        
        return (torch.exp(logprobs) * (logprobs - ref_logprobs) * mask).sum(dim=-1).mean() / token_num * self.kl_div_loss_weight

    def _scores_to_prob(self, scores, ref_scores):
        scores = torch.vstack([torch.unsqueeze(s, dim=0) for s in scores])
        ref_scores = torch.vstack([torch.unsqueeze(s, dim=0) for s in ref_scores])
        min_shape_0 = min(scores.shape[0], ref_scores.shape[0])
        if scores.shape[0] < ref_scores.shape[0]:
            tmax = scores.shape[0] 
            scores = torch.vstack((scores, torch.zeros(ref_scores.shape[0] - tmax, scores.shape[1], scores.shape[2]).to(device)))
        elif ref_scores.shape[0] < scores.shape[0]:
            tmax = ref_scores.shape[0]
            ref_scores = torch.vstack((ref_scores, torch.zeros(scores.shape[0] - tmax, scores.shape[1], scores.shape[2]).to(device)))
        token_num = scores.shape[-1]

        logprobs = F.log_softmax(scores, dim=-1)
        ref_logprobs = F.log_softmax(ref_scores, dim=-1)
        
        return logprobs, ref_logprobs

class PPOTrainer:
    def __init__(self, model, ref_model, optimizer, dataloader, objective):
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.dataloader = dataloader
        
        self.objective = objective

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.model.to(self.device)
        self.ref_model.to(self.device)

        self.loss_history = []

    def train(self, epochs=10, horizon=50, max_length=512):
        update_interval = horizon // self.dataloader.batch_size
        i = 1
        for _ in tqdm(range(epochs)):
            for batch in self.dataloader:
                inputs, label = batch
                tokenized_inputs = self.model.tokenizer(inputs, padding=True, return_tensors='pt').to(self.device)
                label_out = self.model.tokenizer(label, padding=True, return_tensors='pt').to(self.device).input_ids
                action = torch.hstack((torch.zeros(len(inputs), 1).to(torch.device('mps')), label_out)).to(torch.int64)
                
                scores, out = self.model.generate(tokenized_inputs.input_ids, tokenized_inputs.attention_mask, max_length=16)
                ref_scores, _ = self.ref_model.generate(tokenized_inputs.input_ids, tokenized_inputs.attention_mask, max_length=16)

                prediction = []
                for i in range(out.shape[0]):
                    prediction.append(self.model.tokenizer.decode(out[i], skip_special_tokens=True))
                print(prediction)
                
                ppo_iter = PPOIter(tokenized_inputs, action, scores, ref_scores, prediction, label)
                l = self.objective(ppo_iter)
    
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                self.loss_history.append(l.item())

                print(l)
                
                if i % update_interval == 0:
                    self.ref_model = copy.deepcopy(self.model)
                i += 1












