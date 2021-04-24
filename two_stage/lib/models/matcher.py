"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_joints, cost_class: float = 1, cost_coord: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_coord: This is the relative weight of the L1 error of the keypoint coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        self.num_joints = num_joints
        assert cost_class != 0 or cost_coord != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        ## target: [bs, 17, 2]
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
        out_kpt = outputs["pred_coords"]  # [batch_size, num_queries, 2]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[..., :self.num_joints]

        # Compute the L1 cost between keypoints
        cost_kpt = torch.cdist(out_kpt, targets, p=1)  # [B, N, 17]

        # Final cost matrix
        C = self.cost_coord * cost_kpt + self.cost_class * cost_class
        C = C.transpose(1, 2).cpu()  # [B, 17, N]

        indices = [linear_sum_assignment(c) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(num_joints, cost_class=1.0, cost_coord=5.0):
    return HungarianMatcher(num_joints, cost_class=cost_class, cost_coord=cost_coord)