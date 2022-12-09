import torch
from VQ_DR.deezer.models import YoutubeDNN
from VQ_DR.deezer.models.layers import MLP_Layer


class YoutubeDNN_DNN(YoutubeDNN):
    def __init__(self, n_items, item_embedding, user_input_dim, embedding_size, l2_reg, dropout):
        super(YoutubeDNN_DNN, self).__init__(n_items, item_embedding, user_input_dim, embedding_size, l2_reg, dropout)
        self.logits_dnn = MLP_Layer(input_dim=embedding_size*2,
                             output_dim=1,
                             hidden_units=[400, 200, 100],
                             hidden_activations="ReLU",
                             output_activation=None,
                             dropout_rates=dropout,
                             batch_norm=False,
                             use_bias=True)
        
    def get_logits(self, user_embedding, item_embedding):
        interaction_input = torch.concat([user_embedding, item_embedding], dim=-1)  # (B, Num_Neg or 1, D+D)
        logits = self.logits_dnn(interaction_input).squeeze(-1) # (B, Num_Neg or 1)
        return logits
        
    def forward(self, input_dict):
        user_features, pos_items, neg_items = input_dict['user_features'], input_dict['pos_items'], input_dict['neg_items']
        Num_Neg = neg_items.shape[1]
        user_embedding = self.user_encoder(user_features)
        pos_item_embedding = self.item_encoder(pos_items)
        neg_item_embedding = self.item_encoder(neg_items)  # B, Num_Neg, D
        pos_logits = self.get_logits(user_embedding.unsqueeze(1), pos_item_embedding.unsqueeze(1)).squeeze(1)
        neg_logits = self.get_logits(user_embedding.unsqueeze(1).repeat(1, Num_Neg, 1), neg_item_embedding)
        softmax_loss = self.softmax_loss(pos_logits, neg_logits)
        # reg_loss = self.l2_loss(hist_items)
        loss = softmax_loss # + reg_loss
        return loss