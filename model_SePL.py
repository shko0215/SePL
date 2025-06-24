import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Backbone setup
# Here we use ConvNeXt as an example, but you can replace it with any backbone of your choice.
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # You can choose any CNN backbone here.
        self.cnn = models.convnext_tiny(pretrained=True)
        # Remove the classifier part of the CNN
        self.cnn.classifier = nn.Identity()

    def forward(self, x):
        return self.cnn(x)
    

# GatingNetwork for Discriminative Mixture of Experts
# The gating network intakes both image features and anthropometric prior knowledge
# to determine which expert to use for each biomarker prediction.
class GatingNetwork(nn.Module):
    def __init__(self, img_dim=768, anthro_dim=6, num_experts=4):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(img_dim + anthro_dim, num_experts)

    def forward(self, img_features, anthro_preds):
        x = torch.cat([img_features, anthro_preds], dim=1)  # (batch, 768+6) 
        scores = self.fc(x)  # (batch, num_experts)
        gating_probs = F.softmax(scores, dim=1)  # (batch, num_experts)
        return gating_probs


# Attentive weighting for anthropometric prior knowledge
# Regarding the anthropometric prior, learnable weights are applied to each of the 6 dimensions
# to determine their importance in predicting the target biomarkers.
# e.g., if BMI is effective for predicting SBP, a higher weight is assigned to BMI.
class AnthroAttention(nn.Module):
    def __init__(self, anthro_dim=6):
        super(AnthroAttention, self).__init__()
        # Fully connected layer to learn weights for each of the 6 anthropometric dimensions
        self.fc = nn.Linear(anthro_dim, anthro_dim)
        
    def forward(self, anthro_priors):
        # Produce a set of weights for the 6 anthropometric dimensions after passing through fc and softmax
        attn_weights = F.softmax(self.fc(anthro_priors), dim=1)  # (batch, anthro_dim)
        # Elementwise multiplication with the anthropometric prior, yielding effective "selected" values
        selected = anthro_priors * attn_weights
        return selected # (batch, anthro_dim)


# ExpertNetwork for Discriminative Mixture of Experts
# Each expert processes the image features and anthropometric prior knowledge
# to produce a hidden representation that is then used for the final prediction.
# Anthropometric prior is processed through an attentive weighting mechanism to select relevant features.
class ExpertNetwork(nn.Module):
    def __init__(self, img_dim=768, anthro_dim=6, hidden_dim=512):
        super(ExpertNetwork, self).__init__()
        # image feature processing
        self.img_fc = nn.Linear(img_dim, hidden_dim)
        # Anthropometric prior knowledge processing with attentive weighting
        self.anthro_attn = AnthroAttention(anthro_dim)
        self.anthro_fc = nn.Linear(anthro_dim, 12)
        # Multimodal feature fusion for learning diverse feature combinations for 9 different CMR factors
        self.out_fc = nn.Linear(hidden_dim+12, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, img_features, anthro_preds):
        # img_features: (batch, 768)
        # anthro_preds: (batch, 6)
        h_img = self.act(self.img_fc(img_features))  # (batch, hidden_dim)
        # Selectively process anthropometric prior knowledge 
        selected_anthro = self.anthro_attn(anthro_preds)  # (batch, 6)
        h_anthro = self.act(self.anthro_fc(selected_anthro))  # (batch, hidden_dim)
        # Multimodal feature fusion
        combined = torch.cat((h_img, h_anthro), dim=1)
        out = self.act(self.out_fc(combined))  # (batch, hidden_dim)
        return out


# Self-Propagative Multi-Task Learner for predicting diverse CMR factors
class SelfPropagative_MultiTaskLearner(nn.Module):
    def __init__(self, num_experts=4):
        super(SelfPropagative_MultiTaskLearner, self).__init__()
        # (A) Feature extractor backbone (e.g., ConvNeXt, ResNet, etc.)
        self.feature_extractor = Backbone()

        # (B) Anthropometric Prior geads (gender, age, height, weight, bmi, waist)
        self.gender_head = nn.Linear(768, 1)
        self.age_head    = nn.Linear(768, 1)
        self.height_head = nn.Linear(768, 1)
        self.weight_head = nn.Linear(768, 1)
        self.bmi_head    = nn.Linear(768, 1)
        self.waist_head  = nn.Linear(768, 1)

        # Discriminative Mixture of Experts module
        # (C) GatingNetworks for each CMR factor
        self.gating_sbp = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_dbp = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_tchol = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_tg = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_glucose = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_hba1c = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_bun = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_creatinine = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)
        self.gating_uric_acid = GatingNetwork(img_dim=768, anthro_dim=6, num_experts=num_experts)

        # (D) ExpertNetworks
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ExpertNetwork(img_dim=768, anthro_dim=6, hidden_dim=512)
            for _ in range(num_experts)
        ])

        # (E) Self-propagative prediction heads for each CMR factor
        self.sbp_head = nn.Linear(512, 1)
        self.dbp_head = nn.Linear(512, 1)
        self.tchol_head = nn.Linear(512, 1)
        self.tg_head = nn.Linear(512, 1)
        self.glucose_head = nn.Linear(512, 1)
        self.hba1c_head = nn.Linear(512, 1)
        self.bun_head = nn.Linear(512, 1)
        self.creatinine_head = nn.Linear(512, 1)
        self.uric_acid_head = nn.Linear(512, 1)

    def forward(self, x_image):
        # 1) Image features (B, 768)
        img_features = self.feature_extractor(x_image)
        if img_features.dim() > 2: # ConvNeXt output shape (B, 768, 1, 1) to (B, 768)
            img_features = img_features.view(img_features.size(0), -1)

        # 2) Anthropometric prior knowledge prediction (B, 1) for each 
        gender = self.gender_head(img_features)
        age    = self.age_head(img_features)
        height = self.height_head(img_features)
        weight = self.weight_head(img_features)
        bmi    = self.bmi_head(img_features)
        waist  = self.waist_head(img_features)

        # Concatenate anthropometric prior knowledge six (B, 1) to (B, 6)
        anthro_prior = torch.cat([gender, age, height, weight, bmi, waist], dim=1)

        # 3) Gating weights (B, num_experts) from each CMR gate network 
        gating_sbp = self.gating_sbp(img_features, anthro_prior)
        gating_dbp = self.gating_dbp(img_features, anthro_prior)
        gating_tchol = self.gating_tchol(img_features, anthro_prior)
        gating_tg = self.gating_tg(img_features, anthro_prior)
        gating_glucose = self.gating_glucose(img_features, anthro_prior)
        gating_hba1c = self.gating_hba1c(img_features, anthro_prior)
        gating_bun = self.gating_bun(img_features, anthro_prior)
        gating_creatinine = self.gating_creatinine(img_features, anthro_prior)
        gating_uric_acid = self.gating_uric_acid(img_features, anthro_prior)

        # 4) Expert outputs from each expert which learns discriminative multi-modal feature representations
        expert_outputs = [expert(img_features, anthro_prior) for expert in self.experts]

        # 5) Discriminative Fusion: Weighted sum of expert outputs using gating weights
        moe_sbp = sum(gating_sbp[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_dbp = sum(gating_dbp[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_tchol = sum(gating_tchol[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_tg = sum(gating_tg[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_glucose = sum(gating_glucose[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_hba1c = sum(gating_hba1c[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_bun = sum(gating_bun[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_creatinine = sum(gating_creatinine[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))
        moe_uric_acid = sum(gating_uric_acid[:, i].unsqueeze(1) * expert_outputs[i] for i in range(self.num_experts))

        # 6) Self-propagative Prediction (B,1) for each CMR factor
        sbp = self.sbp_head(moe_sbp)
        dbp = self.dbp_head(moe_dbp)
        tchol = self.tchol_head(moe_tchol)
        tg = self.tg_head(moe_tg)
        glucose = self.glucose_head(moe_glucose)
        hba1c = self.hba1c_head(moe_hba1c)
        bun = self.bun_head(moe_bun)
        creatinine = self.creatinine_head(moe_creatinine)
        uric_acid = self.uric_acid_head(moe_uric_acid)
        
        return gender, age, height, weight, bmi, waist, sbp, dbp, tchol, glucose, tg, hba1c, bun, creatinine, uric_acid