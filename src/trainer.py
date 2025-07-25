import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from model import HyperTrip
from rich.progress import track


class ARTrainer:
    def __init__(self, train_dataset, eval_dataset, opt, poi_dis_dict, lr, confidence_score, data,
                 decode_type, train_type, batch_size, num_epochs, d_model, num_encoder_layers,
                 venue_vocab_size, hour_vocab_size, max_length_venue_id, adjacent_matrix, position_matrix,device):

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.opt =opt
        self.poi_dis_dict = poi_dis_dict
        self.data = data
        self.batch_size = batch_size
        self.decode_type = decode_type
        self.train_type = train_type
        self.confidence_score = confidence_score
        self.num_epochs = num_epochs
        self.venue_vocab_size = venue_vocab_size
        self.hour_vocab_size = hour_vocab_size
        self.max_length_venue_id = max_length_venue_id
        self.am = torch.tensor(adjacent_matrix)
        self.pm = torch.tensor(position_matrix)
        self.device = torch.device(device)
        # build dataloader
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                       shuffle=True, drop_last=True)

        self.eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=collate_fn,
                                      drop_last=True)

        # build model(include post-hoc decoding and shift penalty methods)
        self.model = HyperTrip(venue_vocab_size, hour_vocab_size,opt, max_length_venue_id, d_model,
                               n_head=4, num_encoder_layers=num_encoder_layers).to(self.device)
        # for p in self.model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        #     else:
        #         nn.init.uniform_(p)
        # configuration
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding index during loss calculation

    def train(self):

        epoch_f1_score_list = []
        epoch_pairs_f1_list = []
        epoch_repetition_list = []
        epoch_embedding_list =[]
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            total_ids = 0
            correct_predictions = 0
            batch_num = 0
            for masked_padded_venue_ids, masked_padded_hour_ids, padded_venue_ids, padded_hour_ids, _, _ in self.train_loader:

                self.optimizer.zero_grad()
                # counting mask
                src_mask = torch.logical_not(padded_venue_ids.eq(1))
                # for the training methods
                loss = 0
                if self.train_type == 'Normal':

                    venue_output, _, _ = self.model(masked_padded_venue_ids.to(self.device), masked_padded_hour_ids.to(self.device),
                                                 self.am.to(self.device), self.pm.to(self.device), src_mask.to(self.device))  # [b,l,v]

                    venue_output = venue_output.cpu()
                    loss = self.criterion(venue_output.view(-1, self.venue_vocab_size), padded_venue_ids.flatten())

                elif self.train_type == 'Penalty':

                    venue_output, penalty_loss, _ = self.model(masked_padded_venue_ids.to(self.device),
                                                            masked_padded_hour_ids.to(self.device),
                                                            self.am.to(self.device), self.pm.to(self.device), src_mask.to(self.device), batch_num)  # [b,l,v]

                    venue_output = venue_output.cpu()
                    rec_loss = self.criterion(venue_output.view(-1, self.venue_vocab_size), padded_venue_ids.flatten())
                    # the multitask loss
                    loss = rec_loss + penalty_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()

                # Get the predictions for the output venue IDs using Greedy Decoding to verify if the model converges
                _, predicted_ids = torch.max(venue_output, dim=-1)
                predicted_ids.cpu()
                # print results

                # Flatten the predictions and ground truth for comparison
                predicted_ids = predicted_ids.view(-1)
                venue_target = padded_venue_ids.flatten()

                # Exclude the padded values from calculations
                non_padded_indices = venue_target != 0
                venue_target = venue_target[non_padded_indices]
                predicted_ids = predicted_ids[non_padded_indices]

                # Count the number of correctly predicted masked IDs
                total_ids += venue_target.size(0)
                correct_predictions += (predicted_ids == venue_target).sum().item()
                batch_num += 1

            # epoch_loss = total_loss / len(self.train_loader)
            # print(f'train loss: {epoch_loss}')
            # wandb.log({'train loss': epoch_loss, 'epoch': epoch_loss})
            # precision = correct_predictions / total_ids
            # print(f"train precision: {precision}")


            f1, pairs_f1, repetition,embedding_list = self.evaluate()

            epoch_f1_score_list.append(f1)
            epoch_pairs_f1_list.append(pairs_f1)
            epoch_repetition_list.append(repetition)
            epoch_embedding_list.append(embedding_list)
        # calculate results index
        best_index = count_best_index(epoch_f1_score_list, epoch_pairs_f1_list, epoch_repetition_list)

        # pick the best results in each epoch
        return epoch_f1_score_list[best_index], epoch_pairs_f1_list[best_index], epoch_repetition_list[best_index]

    def evaluate(self):
        self.model.eval()

        # for the f1 and pairs f1
        alt_f1_list = []
        alt_pairs_f1_list = []
        embedding_list =[]
        # for the repetition
        repetition_list = []
        batch_num = 0

        with torch.no_grad():
            for masked_padded_venue_ids, masked_padded_hour_ids, padded_venue_ids, padded_hour_ids, _, _ in self.eval_loader:

                # counting mask
                src_mask = torch.logical_not(padded_venue_ids.eq(1))
                # evaluate
                venue_output, _, embedding = self.model(masked_padded_venue_ids.to(self.device), masked_padded_hour_ids.to(self.device),
                                             self.am.to(self.device), self.pm.to(self.device), src_mask.to(self.device), batch_num)
                venue_output = venue_output.cpu()  # [b,l,v]

                # post-hoc decoding methods
                # Greedy Search and Advanced-Greedy
                if self.decode_type == 'Greedy':
                    _, predicted_ids = torch.max(venue_output, dim=-1)
                elif self.decode_type == 'Advanced-Greedy':
                    similarity_ratio, candidate_ids = torch.topk(venue_output, k=venue_output.shape[1], dim=2)
                    predicted_ids = advanced_greedy_recommendation(candidate_ids, similarity_ratio)
                # top-n and top-np search (like LLMs)
                elif self.decode_type == 'Top-N':
                    similarity_ratio, candidate_ids = torch.topk(venue_output, k=venue_output.shape[1], dim=2)
                    predicted_ids = top_n_recommendation(candidate_ids, similarity_ratio,
                                                         confidence=self.confidence_score)  # 1
                elif self.decode_type == 'Top-NP':
                    # each candidate should be considered
                    # [b,l,v]
                    total_similarity_ratio, total_candidate_ids = torch.topk(venue_output, k=venue_output.shape[2],
                                                                             dim=2)
                    # confidence：0.5 threshold：0.8
                    predicted_ids = top_np_recommendation(total_candidate_ids, total_similarity_ratio,
                                                          confidence=self.confidence_score, threshold=0.8)
                else:
                    print("There is no such decoding method!")
                    exit()

                # Flatten the predictions and ground truth for comparison
                predicted_ids = predicted_ids.view(-1)
                venue_target = padded_venue_ids.flatten()

                # Exclude the padded values from calculations
                non_padded_indices = venue_target != 0
                venue_target = venue_target[non_padded_indices]
                predicted_ids = predicted_ids[non_padded_indices]

                # alter the predicted_ids
                alt_predicted_ids = torch.cat((venue_target[:1], predicted_ids[1:-1], venue_target[-1:]), dim=0)

                alt_f1 = f1_score(venue_target, alt_predicted_ids)
                alt_pairs_f1 = pairs_f1_score(venue_target, alt_predicted_ids)
                alt_f1_list.append(alt_f1)
                alt_pairs_f1_list.append(alt_pairs_f1)

                batch_num += 1

                embedding_list.append(embedding)
            repetition_ratio = count_repetition_percentage(predicted_ids)
            repetition_list.append(repetition_ratio)

            # f1 and pairs_f1
            '''f1_score'''
            f1_mean = np.mean(alt_f1_list)
            # print(f"max_f1_score: {max_f1_mean}")
            '''pairs_f1_score'''
            pairs_f1_mean = np.mean(alt_pairs_f1_list)
            # print(f"max_pairs_f1_score: {max_pairs_f1_mean}")

            '''repetition'''
            repetition = np.mean(repetition_list)

            return f1_mean, pairs_f1_mean, repetition,embedding_list


