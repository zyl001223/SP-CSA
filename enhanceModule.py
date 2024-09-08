import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.PReLU, drop=0.2):
        super().__init__()

        if hidden_features is None or out_features is None:
            raise ValueError(
                "Both hidden_features and out_features must be provided.")

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        return query, key, value


class Fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(Fusion, self).__init__()
        self.d_k = hidden_dim
        self.scale = 1.0 / \
            torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, tensor1, tensor2):
        attention_scores = torch.matmul(tensor1, tensor2.t()) * self.scale
        attention_weights = F.softmax(attention_scores, dim=1)
        fusion_tensor = torch.matmul(attention_weights, tensor1)
        fused_tensor = self.bn(tensor1 + fusion_tensor)
        return fused_tensor


class EnhanceCls(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(EnhanceCls, self).__init__()
        self.dalle_adapter = Mlp(input_dim, hidden_dim, hidden_dim)
        self.patch_adapter = Mlp(hidden_dim, hidden_dim, hidden_dim)
        self.fusion = Fusion(hidden_dim)

    def featureWalk(self, prototypes, emb_patch_query, cls_token, selected_k=30, a=1.0):
        prototypes_expanded = prototypes.unsqueeze(1).unsqueeze(
            1).repeat(1, 75, 196, 1)  # (5, 75, 196, 384)
        emb_patch_query_expanded = emb_patch_query.unsqueeze(
            0).repeat(5, 1, 1, 1)  # (5, 75, 196, 384)
        cosine_sim = F.cosine_similarity(
            emb_patch_query_expanded, prototypes_expanded, dim=3)  # (5, 75, 196)
        softmax_weights = F.softmax(cosine_sim, dim=2)  # (5, 75, 196)
        weighted_sums = torch.zeros(
            5, 75, 384, device=emb_patch_query.device)
        for k in range(5):
            for i in range(75):

                weights = softmax_weights[k, i]  # (196,)

                _, top_n_indices = weights.topk(selected_k, dim=0)  # (10,)

                top10_patches = emb_patch_query[i, top_n_indices]  # (10, 384)

                top10_weights = weights[top_n_indices]  # (10,)

                weighted_sum = torch.sum(
                    top10_weights.unsqueeze(1) * top10_patches, dim=0)

                weighted_sums[k, i] = weighted_sum

        cls_token_expanded = cls_token.unsqueeze(
            0).repeat(5, 1, 1)  # (5, 75, 384)
        cls_token_expanded = cls_token_expanded*2
        cls_with_weighted_sums = cls_token_expanded + \
            a*weighted_sums  # (5, 75, 384)
        return cls_with_weighted_sums

    def enhanceCls(self, cls_token, patch_token, k=30):

        batch_size, feature_dim = cls_token.size()
        num_patches = patch_token.size(1)

        cls_token_expanded = cls_token.unsqueeze(1).repeat(
            1, num_patches, 1)  # (batch_size, num_patches, feature_dim)

        # (batch_size, num_patches)
        similarity = torch.sum(patch_token * cls_token_expanded, dim=2)

        all_similarities = torch.sum(
            similarity, dim=1, keepdim=True)  # (batch_size, 1)

        weighted_similarity = similarity / \
            all_similarities  # (batch_size, num_patches)

        topk_values, topk_indices = weighted_similarity.topk(
            k, dim=1)  # (batch_size, k)

        selected_patches = torch.gather(
            patch_token, 1, topk_indices.unsqueeze(-1).repeat(1, 1, feature_dim))

        weighted_patches = selected_patches * topk_values.unsqueeze(-1)
        enhanced_features = torch.sum(weighted_patches, dim=1)

        enhanced_cls_token = cls_token + enhanced_features

        return enhanced_cls_token

    def enhancePrototypes(self, cls_token, patch_token, k=30,  a=1.0):

        enhanced_cls_token = torch.clone(cls_token)
        enhanced_cls_token = enhanced_cls_token * 2
        total_class_number, class_index, feature_dim = cls_token.size()
        num_patches = patch_token.size(2)

        expanded_cls_token = cls_token.unsqueeze(
            2).expand(-1, -1, num_patches, -1)

        distances = (expanded_cls_token - patch_token).norm(dim=3)

        other_distances = torch.zeros_like(distances)

        for cls_num_idx in range(total_class_number):

            mask = torch.ones(total_class_number,
                              dtype=torch.bool, device=cls_token.device)
            mask[cls_num_idx] = False

            other_cls_distances = distances[mask, :, :]

            other_distances[cls_num_idx, :, :] = other_cls_distances.sum(dim=0)[
                0]

        similarity = distances / (other_distances + 1e-6)

        topk_values, topk_idx = similarity.topk(k, dim=2)

        for cls_num_idx in range(total_class_number):
            for cls_idx in range(class_index):

                selected_patches = patch_token[cls_num_idx,
                                               cls_idx, topk_idx[cls_num_idx, cls_idx]]

                enhanced_cls_token[cls_num_idx,
                                   cls_idx] += a * selected_patches.mean(dim=0)

        return enhanced_cls_token

    def forward(self, support_set_vectors, query_set_vectors, dalle_emb_support, emb_patch_support, emb_patch_query, dalle_patch_embedding):

        support_set_vectors = support_set_vectors.squeeze(
            1).reshape(5, -1, 384)  # torch.Size([5，5, 384])
        query_set_vectors = query_set_vectors.squeeze(
            1).reshape(5, -1, 384)    # torch.Size([5，15, 384])
        dalle_emb_support = dalle_emb_support.squeeze(
            1).reshape(5, -1, 384)   # torch.Size([5，5, 384])

        dalle_emb_support = self.dalle_adapter(
            dalle_emb_support.reshape(-1, 384)).reshape(5, -1, 384)  # torch.Size([5,5, 384])
        dalle_patch_embedding = self.dalle_adapter(
            dalle_patch_embedding.reshape(-1, 384)).reshape(-1, 196, 384)  # torch.Size([25, 196, 384])

        emb_patch_support = emb_patch_support + self.patch_adapter(
            emb_patch_support.reshape(-1, 384)).reshape(-1, 196, 384)  # torch.Size([25, 196, 384])
        emb_patch_query = emb_patch_query + self.patch_adapter(
            emb_patch_query.reshape(-1, 384)).reshape(-1, 196, 384)  # torch.Size([75, 196, 384])
        dalle_patch_embedding = dalle_patch_embedding + self.patch_adapter(
            dalle_patch_embedding.reshape(-1, 384)).reshape(-1, 196, 384)  # torch.Size([25, 196, 384])

        prototypes = torch.cat(
            (support_set_vectors, dalle_emb_support), dim=1).mean(dim=1).squeeze(1)

        enhance_prototypes = torch.cat(
            (self.enhancePrototypes(support_set_vectors, emb_patch_support.reshape(5, -1, 196, 384)), self.enhancePrototypes(dalle_emb_support, dalle_patch_embedding.reshape(5, -1, 196, 384))), dim=1).mean(dim=1).squeeze(1)

        cls_with_weighted_sums = self.featureWalk(
            enhance_prototypes, emb_patch_query, query_set_vectors.reshape(-1, 384))

        return enhance_prototypes, cls_with_weighted_sums
