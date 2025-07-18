import torch
import ctypes
import math
from typing import List, Dict, Tuple, Any

from problem import Problem


class n_body_simulation(Problem):
    """N-body gravitational simulation problem."""
    
    def __init__(self):
        super().__init__(
            name="n-body-simulation"
        )
    
    def reference_solution(self, positions: torch.Tensor, velocities: torch.Tensor, masses: torch.Tensor, n_bodies: int, dt: float, G: float) -> torch.Tensor:
        """
        PyTorch implementation of N-body simulation (one timestep).
        
        Args:
            positions: Current positions (n_bodies x 3)
            velocities: Current velocities (n_bodies x 3)
            masses: Masses of bodies (n_bodies,)
            n_bodies: Number of bodies
            dt: Time step
            G: Gravitational constant
            
        Returns:
            New positions after one timestep (n_bodies x 3)
        """
        with torch.no_grad():
            # Calculate forces between all pairs of bodies
            forces = torch.zeros_like(positions)
            
            for i in range(n_bodies):
                for j in range(n_bodies):
                    if i != j:
                        # Vector from body i to body j
                        r_vec = positions[j] - positions[i]
                        r_distance = torch.norm(r_vec)
                        
                        # Avoid division by zero with softening parameter
                        epsilon = 1e-3
                        r_distance = torch.clamp(r_distance, min=epsilon)
                        
                        # Gravitational force magnitude: F = G * m1 * m2 / r^2
                        force_magnitude = G * masses[i] * masses[j] / (r_distance * r_distance)
                        
                        # Force direction (unit vector)
                        force_direction = r_vec / r_distance
                        
                        # Add force to body i
                        forces[i] += force_magnitude * force_direction
            
            # Update velocities: v_new = v_old + (F/m) * dt
            new_velocities = velocities + (forces / masses.unsqueeze(1)) * dt
            
            # Update positions: x_new = x_old + v_new * dt
            new_positions = positions + new_velocities * dt
            
            return new_positions
    
    def generate_test_cases(self, dtype: torch.dtype) -> List[Dict[str, Any]]:
        """
        Generate test cases for N-body simulation.
        
        Returns:
            List of test case dictionaries with varying body counts
        """
        test_cases = [
            {
                "name": "small_system_100",
                "n_bodies": 100,
                "dt": 0.01,
                "G": 1.0,
                "create_inputs": lambda: self._create_random_system(100, dtype)
            },
            {
                "name": "medium_system_1000",
                "n_bodies": 1000,
                "dt": 0.001,
                "G": 1.0,
                "create_inputs": lambda: self._create_random_system(1000, dtype)
            },
            {
                "name": "large_system_5000",
                "n_bodies": 5000,
                "dt": 0.0005,
                "G": 1.0,
                "create_inputs": lambda: self._create_random_system(5000, dtype)
            },
            {
                "name": "galaxy_system_10000",
                "n_bodies": 10000,
                "dt": 0.0001,
                "G": 1.0,
                "create_inputs": lambda: self._create_galaxy_system(10000, dtype)
            }
        ]
        
        return test_cases
    
    def _create_random_system(self, n_bodies: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
        """Helper to create a random system of bodies."""
        # Random positions in a cube
        positions = torch.randn(n_bodies, 3, device="cuda", dtype=dtype) * 10.0
        
        # Random velocities
        velocities = torch.randn(n_bodies, 3, device="cuda", dtype=dtype) * 0.1
        
        # Random masses (positive)
        masses = torch.rand(n_bodies, device="cuda", dtype=dtype) * 2.0 + 0.5
        
        return (positions, velocities, masses, n_bodies, 0.01, 1.0)
    
    def _create_galaxy_system(self, n_bodies: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
        """Helper to create a galaxy-like system with central mass."""
        # Central massive body
        positions = torch.zeros(n_bodies, 3, device="cuda", dtype=dtype)
        velocities = torch.zeros(n_bodies, 3, device="cuda", dtype=dtype)
        masses = torch.ones(n_bodies, device="cuda", dtype=dtype)
        
        # Central body
        masses[0] = 1000.0  # Very massive center
        
        # Surrounding bodies in orbital configuration
        for i in range(1, n_bodies):
            # Random distance from center
            r = torch.rand(1, device="cuda", dtype=dtype) * 20.0 + 5.0
            
            # Random angles
            theta = torch.rand(1, device="cuda", dtype=dtype) * 2 * math.pi
            phi = torch.rand(1, device="cuda", dtype=dtype) * math.pi
            
            # Spherical to cartesian
            positions[i, 0] = r * torch.sin(phi) * torch.cos(theta)
            positions[i, 1] = r * torch.sin(phi) * torch.sin(theta)
            positions[i, 2] = r * torch.cos(phi)
            
            # Orbital velocity (simplified)
            v_mag = torch.sqrt(masses[0] / r) * 0.1
            velocities[i, 0] = -v_mag * torch.sin(theta)
            velocities[i, 1] = v_mag * torch.cos(theta)
            velocities[i, 2] = 0.0
        
        return (positions, velocities, masses, n_bodies, 0.0001, 1.0)
    
    def generate_sample(self, dtype: torch.dtype = torch.float32) -> Dict[str, Any]:
        """
        Generate a single sample test case for debugging.
        
        Returns:
            A dictionary containing a single test case
        """
        return {
            "name": "Sample (n=10)",
            "n_bodies": 10,
            "dt": 0.1,
            "G": 1.0,
            "create_inputs": lambda: self._create_sample_system(dtype)
        }
    
    def _create_sample_system(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, float]:
        """Create a small sample system for debugging."""
        # Simple 10-body system
        positions = torch.tensor([
            [0.0, 0.0, 0.0],    # Central body
            [1.0, 0.0, 0.0],    # Body 1
            [0.0, 1.0, 0.0],    # Body 2
            [-1.0, 0.0, 0.0],   # Body 3
            [0.0, -1.0, 0.0],   # Body 4
            [2.0, 0.0, 0.0],    # Body 5
            [0.0, 2.0, 0.0],    # Body 6
            [-2.0, 0.0, 0.0],   # Body 7
            [0.0, -2.0, 0.0],   # Body 8
            [1.0, 1.0, 0.0]     # Body 9
        ], device="cuda", dtype=dtype)
        
        velocities = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [-0.05, 0.0, 0.0],
            [0.0, -0.05, 0.0],
            [0.05, 0.0, 0.0],
            [-0.07, 0.07, 0.0]
        ], device="cuda", dtype=dtype)
        
        masses = torch.tensor([
            10.0,  # Central body
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ], device="cuda", dtype=dtype)
        
        return (positions, velocities, masses, 10, 0.1, 1.0)

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor, dtype: torch.dtype) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the N-body simulation result is correct.
        
        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution
            
        Returns:
            Tuple of (is_correct, debug_info)
        """
        # Physics simulations can have numerical precision issues
        is_close = torch.allclose(actual_output, expected_output, rtol=1e-4, atol=1e-3)
        
        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()
            
            # Check individual body differences
            per_body_diff = torch.norm(diff, dim=1)
            worst_body = torch.argmax(per_body_diff).item()
            
            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "worst_body_index": worst_body,
                "worst_body_error": per_body_diff[worst_body].item(),
                "expected_range": [torch.min(expected_output).item(), torch.max(expected_output).item()],
                "actual_range": [torch.min(actual_output).item(), torch.max(actual_output).item()]
            }
        
        return is_close, debug_info
    
    def get_function_signature(self) -> Dict[str, Any]:
        """
        Get the function signature for the N-body simulation solution.
        
        Returns:
            Dictionary with argtypes and restype for ctypes
        """
        return {
            "argtypes": [
                ctypes.POINTER(ctypes.c_float),  # positions (input)
                ctypes.POINTER(ctypes.c_float),  # velocities (input)
                ctypes.POINTER(ctypes.c_float),  # masses (input)
                ctypes.POINTER(ctypes.c_float),  # new_positions (output)
                ctypes.c_size_t,                 # n_bodies
                ctypes.c_float,                  # dt
                ctypes.c_float                   # G
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
        n_bodies = test_case["n_bodies"]
        
        # For each body i, calculate forces from all other bodies j (i != j):
        # - Vector difference: 3 subtractions
        # - Distance calculation: 3 squares + 2 additions + 1 sqrt = 6 operations
        # - Force magnitude: 3 multiplications + 1 division = 4 operations
        # - Force direction: 3 divisions = 3 operations
        # - Force accumulation: 3 additions = 3 operations
        # Total per pair: 3 + 6 + 4 + 3 + 3 = 19 operations
        # Total pairs: n_bodies * (n_bodies - 1)
        force_calculation_ops = n_bodies * (n_bodies - 1) * 19
        
        # Velocity update: n_bodies * 3 * 3 = 9 * n_bodies operations
        # (force/mass * dt for each component)
        velocity_update_ops = 9 * n_bodies
        
        # Position update: n_bodies * 3 * 2 = 6 * n_bodies operations
        # (position + velocity * dt for each component)
        position_update_ops = 6 * n_bodies
        
        return force_calculation_ops + velocity_update_ops + position_update_ops
    
    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.
        
        Args:
            test_case: The test case dictionary
            
        Returns:
            List containing the simulation parameters
        """
        return [
            test_case["n_bodies"],
            test_case["dt"],
            test_case["G"]
        ] 