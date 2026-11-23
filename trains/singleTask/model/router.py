import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitRouter(nn.Module):
    def __init__(self, logit_dim, alpha=5.0, n_modal=4, tau=1.0, lambda_mea=0.1, beta=1.0,
                 hyp_c=0.05, hyp_r=2.3, alpha_d=0.5, tau_r=0.1, momentum=0.999):
        super().__init__()
        self.logit_dim = logit_dim
        self.alpha = alpha
        self.n_modal = n_modal
        self.tau = tau
        self.lambda_mea = lambda_mea
        self.beta = beta

        # hyperbolic params
        self.hyp_c = hyp_c
        self.hyp_r = hyp_r
        self.hyp_eps = 1e-8
        self.alpha_d = alpha_d      
        self.tau_r = tau_r         
        self.momentum = momentum

        euclid_prototypes = torch.diag(torch.full((logit_dim,), alpha, dtype=torch.float32))
        clipped_prototypes = self._clip_feature(euclid_prototypes)
        hyp_prototypes = self._exponential_map(clipped_prototypes)
        proto_tensor = self._safe_projection(hyp_prototypes).detach()
        self.register_buffer('prototypes', proto_tensor)

    def _clip_feature(self, z):
        z_norm = torch.norm(z, p=2, dim=-1, keepdim=True)
        clip_factor = torch.min(torch.ones_like(z_norm), self.hyp_r / (z_norm + self.hyp_eps))
        return z * clip_factor

    def _exponential_map(self, z):
        z_norm = torch.norm(z, p=2, dim=-1, keepdim=True)
        if (z_norm < self.hyp_eps).all():
            return z
        sqrt_c = torch.sqrt(torch.tensor(self.hyp_c, device=z.device))
        scale = torch.tanh(sqrt_c * z_norm) / (sqrt_c * z_norm)
        return scale * z

    def _mobius_subtract(self, a, b):
        a_norm_sq = torch.norm(a, p=2, dim=-1, keepdim=True) ** 2
        b_norm_sq = torch.norm(b, p=2, dim=-1, keepdim=True) ** 2
        ab_dot = torch.sum(a * b, dim=-1, keepdim=True)
        c = self.hyp_c
        numerator = (1 - 2 * c * ab_dot + c * b_norm_sq) * a + (1 - c * a_norm_sq) * (-b)
        denominator = 1 - 2 * c * ab_dot + c ** 2 * a_norm_sq * b_norm_sq
        return numerator / (denominator + self.hyp_eps)

    def _hyperbolic_distance(self, a, b):
        # a, b: (..., dim)
        mobius_diff = self._mobius_subtract(a, b)
        diff_norm = torch.norm(mobius_diff, p=2, dim=-1)
        sqrt_c = torch.sqrt(torch.tensor(self.hyp_c, device=a.device))
        tanh_arg = sqrt_c * diff_norm
        tanh_arg = torch.clamp(tanh_arg, max=1 - self.hyp_eps)
        dist = (2 / sqrt_c) * 0.5 * torch.log((1 + tanh_arg) / (1 - tanh_arg + self.hyp_eps))
        return dist

    def _safe_projection(self, z):
        z_norm = torch.norm(z, p=2, dim=-1, keepdim=True)
        max_norm = (1 - self.hyp_eps) / torch.sqrt(torch.tensor(self.hyp_c, device=z.device))
        scale = torch.min(torch.ones_like(z_norm), max_norm / (z_norm + self.hyp_eps))
        return z * scale

    def _riemannian_gradient_hook(self, grad, z_norm_sq):
        scale = ((1 - self.hyp_c * z_norm_sq) ** 2) / 4.0
        return grad * scale

    def _update_prototypes_ema(self, hyp_z_list, y):
        with torch.no_grad():
            all_features = torch.cat(hyp_z_list, dim=0).detach()
            all_labels = y.repeat(len(hyp_z_list))
            
            unique_classes = torch.unique(all_labels)
            for k in unique_classes:
                class_mask = (all_labels == k)
                class_features = all_features[class_mask]
                batch_centroid = torch.mean(class_features, dim=0)
                
                new_proto = self.momentum * self.prototypes[k] + (1 - self.momentum) * batch_centroid
                self.prototypes[k].data.copy_(self._safe_projection(new_proto))

    def forward(self, z1, z2, z3, z4, y=None):
        z_list = [z1, z2, z3, z4]
        batch_size = z_list[0].shape[0]
        device = z_list[0].device

        H_list = []
        hyp_z_list = []
        for z in z_list:
            z_clipped = self._clip_feature(z)
            z_hyp = self._exponential_map(z_clipped)
            z_hyp = self._safe_projection(z_hyp)
            
            if self.training:
                z_norm_sq = torch.sum(z_hyp ** 2, dim=-1, keepdim=True)
                z_hyp.register_hook(lambda grad, s=z_norm_sq: self._riemannian_gradient_hook(grad, s))
            hyp_z_list.append(z_hyp) # 收集用于后续计算

            d_list = []
            curr_prototypes = self.prototypes.detach()
            
            for proto in curr_prototypes:
                proto_expand = proto.unsqueeze(0).expand(batch_size, -1)
                d_tmp = self._hyperbolic_distance(z_hyp, proto_expand) 
                d_list.append(d_tmp.unsqueeze(1))
            d = torch.cat(d_list, dim=1)  # (batch, K)

            d_recip = 1.0 / (d + self.hyp_eps)
            d_l1 = torch.norm(d_recip, p=1, dim=1, keepdim=True)
            hat_d = d_recip / d_l1
            H = -torch.sum(hat_d * torch.log(hat_d + self.hyp_eps), dim=1)
            H_list.append(H)

        H_stack = torch.stack(H_list, dim=1)  # (batch, n_modal)
        W = F.softmax(-H_stack / self.tau, dim=1)

        if not self.training or y is None:
            return W
        
        self._update_prototypes_ema(hyp_z_list, y)

        modal_losses = []
        for hyp_z in hyp_z_list:
            all_prototypes = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1) 

            hyp_z_norm = F.normalize(hyp_z, p=2, dim=-1)
            proto_norm = F.normalize(all_prototypes, p=2, dim=-1)
            S_a = torch.einsum('bd,bkd->bk', hyp_z_norm, proto_norm)

            S_d = torch.zeros((batch_size, self.logit_dim), device=device)
            for i in range(batch_size):
                dists = self._hyperbolic_distance(hyp_z[i].unsqueeze(0), all_prototypes[i])
                S_d[i] = -dists

            L_dis = 0.0
            L_ang = 0.0

            for i in range(batch_size):
                true_class = y[i]
                N_i = 1
                pos_S_d = S_d[i, true_class].unsqueeze(0)
                pos_S_a = S_a[i, true_class].unsqueeze(0)
                
                neg_mask = torch.arange(self.logit_dim, device=device) != true_class
                neg_S_d = S_d[i, neg_mask]
                neg_S_a = S_a[i, neg_mask]

                numerator_dis = torch.exp(pos_S_d / self.tau_r).sum()
                denom_dis = torch.exp(neg_S_d / self.tau_r).sum() + numerator_dis
                L_dis_i = -torch.log(numerator_dis / (denom_dis + self.hyp_eps))/ N_i
                L_dis += L_dis_i

                numerator_ang = torch.exp(pos_S_a / self.tau_r).sum()
                denom_ang = torch.exp(neg_S_a / self.tau_r).sum() + numerator_ang
                L_ang_i = -torch.log(numerator_ang / (denom_ang + self.hyp_eps))/ N_i
                L_ang += L_ang_i

            L_dis_avg = L_dis / batch_size
            L_ang_avg = L_ang / batch_size
            modal_loss = self.alpha_d * L_dis_avg + (1 - self.alpha_d) * L_ang_avg
            modal_losses.append(modal_loss)

        L_joint = torch.mean(torch.stack(modal_losses, dim=0))
        return W, L_joint
