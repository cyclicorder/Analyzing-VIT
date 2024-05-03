import torch
import torch.nn.functional as F

class PGDEmbeddingOptimizer:
    """ Class for optimizing image embeddings using projected gradient descent.
    Attributes:
    model (torch.nn.Module): Neural network model for generating embeddings.
    learning_rate (float): Learning rate for gradient descent.
    """
    def __init__(self, model, learning_rate):
        """ Initializes the EmbeddingOptimizer with a model and learning rate.
        Args:
        model (torch.nn.Module): Model for generating embeddings.
        learning_rate (float): Learning rate for gradient descent.
        """
        self.model = model
        self.lr = learning_rate

    def optimize_embeddings_pgd(self, cur_input, target_emb, epsilon, l2_dist_threshold, cosine_sim_threshold):
        """ Adjusts initial input to match target embedding using Projected Gradient Descent.
        Args:
        cur_input (torch.Tensor): Input tensor to be optimized.
        target_emb (torch.Tensor): Target embedding to match.
        epsilon (float): Epsilon value for controlling perturbations in PGD.
        l2_dist_threshold (float): Threshold for squared L2 distance.
        cosine_sim_threshold (float): Threshold for cosine similarity.
        Returns:
        torch.Tensor: Optimized input tensor.
        list: L1 distances over iterations.
        list: Cosine similarities over iterations.
        list: Losses over iterations.
        """
        org_input = cur_input.clone()
        squared_l2_distance = float('inf')
        cosine_sim_arr = []
        loss_arr = []
        l1_dist_arr = []
        exit_counter = 0
        squared_l2_distance_arr = []
        iteration_count = 0
        cosine_sim = 0
        squared_l2_distance = float('inf')
        while squared_l2_distance >= l2_dist_threshold or cosine_sim <= cosine_sim_threshold:
            cur_input = cur_input.clone().detach().requires_grad_(True)
            outputs = self.model(pixel_values=cur_input, output_hidden_states=True)
            cur_emb = outputs.hidden_states[-1][:, 0, :]  # embedding of CLS token
            loss = F.mse_loss(target_emb, cur_emb)
            loss_arr.append(loss.item())
            cur_input.grad = None
            loss.backward(retain_graph=True)
            grad = cur_input.grad
            updated_input = cur_input - self.lr * grad
            projected_input = cur_input + torch.clamp(updated_input - cur_input, -epsilon, epsilon)
            with torch.no_grad():
                updated_outputs = self.model(pixel_values=projected_input, output_hidden_states=True)
                updated_emb = updated_outputs.hidden_states[-1][:, 0, :]
                squared_l2_distance = torch.sum((target_emb - updated_emb) ** 2, dim=1).mean().item()
                updated_l1_dist = torch.sum(torch.abs(projected_input.detach() - org_input)).item()
                cosine_sim = F.cosine_similarity(target_emb.unsqueeze(0), updated_emb.unsqueeze(0), dim=1).mean()
                print(squared_l2_distance, ' - ', cosine_sim)
                l1_dist_arr.append(updated_l1_dist)
                cosine_sim_arr.append(cosine_sim.detach().cpu().item())
                squared_l2_distance_arr.append(squared_l2_distance)
            cur_input = projected_input
            iteration_count += 1
        return cur_input.detach(), l1_dist_arr, cosine_sim_arr, loss_arr, squared_l2_distance_arr, iteration_count, cosine_sim, squared_l2_distance