import torch
import ctypes
import math
from typing import List, Dict, Tuple, Any

from problem import Problem


class k_means_clustering(Problem):
    """K-means clustering problem."""
    
    def __init__(self):
        super().__init__(
            name="k-means-clustering"
        )
    
    def reference_solution(self, points: torch.Tensor, centroids: torch.Tensor, n_points: int, n_features: int, k: int, max_iter: int) -> torch.Tensor:
        """
        PyTorch implementation of K-means clustering.
        
        Args:
            points: Input data points (n_points x n_features)
            centroids: Initial centroids (k x n_features)
            n_points: Number of data points
            n_features: Number of features per point
            k: Number of clusters
            max_iter: Maximum number of iterations
            
        Returns:
            Final centroids after convergence (k x n_features)
        """
        with torch.no_grad():
            current_centroids = centroids.clone()
            
            for iteration in range(max_iter):
                # Assign points to nearest centroids
                distances = torch.cdist(points, current_centroids, p=2)  # (n_points, k)
                assignments = torch.argmin(distances, dim=1)  # (n_points,)
                
                # Update centroids
                new_centroids = torch.zeros_like(current_centroids)
                for cluster_id in range(k):
                    mask = (assignments == cluster_id)
                    if torch.sum(mask) > 0:
                        new_centroids[cluster_id] = torch.mean(points[mask], dim=0)
                    else:
                        # Keep old centroid if no points assigned
                        new_centroids[cluster_id] = current_centroids[cluster_id]
                
                # Check for convergence
                if torch.allclose(current_centroids, new_centroids, rtol=1e-6, atol=1e-6):
                    break
                    
                current_centroids = new_centroids
            
            return current_centroids
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for K-means clustering.
        
        Returns:
            List of test case dictionaries with varying data sizes and dimensions
        """
        test_cases = [
            {
                "name": "small_2d_3clusters",
                "n_points": 1000,
                "n_features": 2,
                "k": 3,
                "max_iter": 100,
                "create_inputs": lambda: self._create_clustered_data(1000, 2, 3, dtype)
            },
            {
                "name": "medium_5d_5clusters",
                "n_points": 10000,
                "n_features": 5,
                "k": 5,
                "max_iter": 100,
                "create_inputs": lambda: self._create_clustered_data(10000, 5, 5, dtype)
            },
            {
                "name": "large_10d_8clusters",
                "n_points": 50000,
                "n_features": 10,
                "k": 8,
                "max_iter": 100,
                "create_inputs": lambda: self._create_clustered_data(50000, 10, 8, dtype)
            },
            {
                "name": "xlarge_20d_10clusters",
                "n_points": 100000,
                "n_features": 20,
                "k": 10,
                "max_iter": 100,
                "create_inputs": lambda: self._create_clustered_data(100000, 20, 10, dtype)
            }
        ]
        
        return test_cases
    
    def _create_clustered_data(self, n_points: int, n_features: int, k: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """Helper to create synthetic clustered data."""
        # Generate k cluster centers
        cluster_centers = torch.randn(k, n_features, device="cuda", dtype=dtype) * 10.0
        
        # Generate points around each cluster
        points = torch.zeros(n_points, n_features, device="cuda", dtype=dtype)
        points_per_cluster = n_points // k
        
        for i in range(k):
            start_idx = i * points_per_cluster
            end_idx = (i + 1) * points_per_cluster if i < k - 1 else n_points
            
            # Generate points around cluster center with some noise
            cluster_points = cluster_centers[i].unsqueeze(0) + torch.randn(end_idx - start_idx, n_features, device="cuda", dtype=dtype) * 2.0
            points[start_idx:end_idx] = cluster_points
        
        # Randomly initialize centroids (not at true centers)
        initial_centroids = torch.randn(k, n_features, device="cuda", dtype=dtype) * 5.0
        
        return (points, initial_centroids, n_points, n_features, k, 100)
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging.
        
        Returns:
            A dictionary containing a single test case
        """
        return {
            "name": "Sample (n=100, d=2, k=3)",
            "n_points": 100,
            "n_features": 2,
            "k": 3,
            "max_iter": 50,
            "create_inputs": lambda: self._create_sample_data(dtype)
        }
    
    def _create_sample_data(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
        """Create a small sample dataset for debugging."""
        # Create 3 clear clusters in 2D
        torch.manual_seed(42)
        
        # Cluster 1: around (0, 0)
        cluster1 = torch.randn(30, 2, device="cuda", dtype=dtype) * 0.5
        
        # Cluster 2: around (5, 5)
        cluster2 = torch.randn(35, 2, device="cuda", dtype=dtype) * 0.5 + torch.tensor([5.0, 5.0], device="cuda", dtype=dtype)
        
        # Cluster 3: around (-3, 4)
        cluster3 = torch.randn(35, 2, device="cuda", dtype=dtype) * 0.5 + torch.tensor([-3.0, 4.0], device="cuda", dtype=dtype)
        
        points = torch.cat([cluster1, cluster2, cluster3], dim=0)
        
        # Initial centroids (not at true centers)
        centroids = torch.tensor([
            [1.0, 1.0],
            [3.0, 3.0],
            [-1.0, 2.0]
        ], device="cuda", dtype=dtype)
        
        return (points, centroids, 100, 2, 3, 50)

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the K-means clustering result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # K-means can converge to different local minima, so we need to check
        # if the centroids are equivalent up to permutation
        is_close = self._check_centroids_equivalent(expected_output, actual_output)
        
        debug_info = {}
        if not is_close:
            # Calculate distances between each expected and actual centroid
            distances = torch.cdist(expected_output, actual_output, p=2)
            min_distances = torch.min(distances, dim=1)[0]
            
            debug_info = {
                "max_centroid_distance": torch.max(min_distances).item(),
                "mean_centroid_distance": torch.mean(min_distances).item(),
                "expected_centroids": expected_output.cpu().numpy().tolist(),
                "actual_centroids": actual_output.cpu().numpy().tolist()
            }
        
        return is_close, debug_info
    
    def _check_centroids_equivalent(self, expected: torch.Tensor, actual: torch.Tensor, tol: float = 1e-2) -> bool:
        """Check if two sets of centroids are equivalent up to permutation."""
        k = expected.shape[0]
        
        # Try all permutations to find the best match
        from itertools import permutations
        
        best_match = False
        for perm in permutations(range(k)):
            permuted_actual = actual[list(perm)]
            if torch.allclose(expected, permuted_actual, rtol=tol, atol=tol):
                best_match = True
                break
        
        return best_match
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the K-means clustering solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # points
                ctypes.POINTER(ctypes.c_float),  # centroids (input/output)
                ctypes.c_size_t,                 # n_points
                ctypes.c_size_t,                 # n_features
                ctypes.c_size_t,                 # k
                ctypes.c_int                     # max_iter
            ],
            "restype": None
        }
    
    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.
        
        Args:
            test_case: The test case dictionary

        IMPORTANT: Comments are required. Outline the FLOPs calculation.
            
        Returns:
            Number of floating point operations
        """
        n_points = test_case["n_points"]
        n_features = test_case["n_features"]
        k = test_case["k"]
        max_iter = test_case["max_iter"]
        
        # Per iteration:
        # 1. Distance calculation: n_points * k * (n_features subtractions + n_features squares + 1 sqrt)
        #    ≈ n_points * k * (2 * n_features + 1)
        # 2. Assignment: n_points * k comparisons (minimal cost)
        # 3. Centroid update: k * n_features * (n_points/k) operations on average
        #    = n_points * n_features operations
        # Total per iteration: n_points * k * (2 * n_features + 1) + n_points * n_features
        # Total: max_iter * [n_points * k * (2 * n_features + 1) + n_points * n_features]
        per_iter_flops = n_points * k * (2 * n_features + 1) + n_points * n_features
        return max_iter * per_iter_flops
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the clustering parameters
        """
        return [
            test_case["n_points"],
            test_case["n_features"],
            test_case["k"],
            test_case["max_iter"]
        ] 