import numpy as np
import scipy.stats


class SimBboxes:
    """Class that implements SimBboxes criterion."""

    def __init__(self, variant):
        self.variant = variant

    def compute_criterion(self, a_map, bounding_boxes):
        """Compute the sum of activation values within and outside the bounding boxes.

        Args:
            a_map (numpy.ndarray): The saliency map with activation values for each pixel.
            bounding_boxes (list of dict): A list of bounding box coordinates, where each box is represented
                                        as a dictionary with 'x_min', 'y_min', 'x_max', 'y_max'.

        Returns:
            tuple: (sum_inside_bboxes, sum_outside_bboxes) - The sum of activation values inside and outside the bounding boxes.
        """
        # Initialize masks for inside and outside bounding boxes
        inside_bbox_mask = np.zeros_like(a_map, dtype=bool)

        a_map = np.clip(a_map, 0, 1)

        # Loop over each bounding box
        for box in bounding_boxes:
            x_min, y_min = int(box["x_min"]), int(box["y_min"])
            x_max, y_max = int(box["x_max"]), int(box["y_max"])

            # Create a mask for the current bounding box
            inside_bbox_mask[y_min:y_max, x_min:x_max] = True

        if self.variant == "in":
            # Sum activations inside bounding boxes
            return np.sum(a_map[inside_bbox_mask])

        if self.variant == "out":
            # Sum activations outside bounding boxes (where mask is False)
            return np.sum(a_map[~inside_bbox_mask])

        if self.variant == "all":
            return np.sum(a_map)

        if self.variant == "entropy":
            # Flatten the saliency map to a 1D array
            flattened_map = a_map.flatten()

            # Remove zero values to avoid log(0) issues (which would result in NaN)
            flattened_map = flattened_map[flattened_map > 0]

            # Normalize the saliency map to get probabilities (sum should be 1)
            prob_map = flattened_map / np.sum(flattened_map)

            # Compute entropy using the formula -sum(p(x) * log(p(x)))
            return scipy.stats.entropy(prob_map)

        ValueError("Invalid variant. Choose 'in', 'out', 'all' or 'entropy'.")
        return None
