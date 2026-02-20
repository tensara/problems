import torch
from typing import List, Dict, Tuple, Any

from problem import Problem

class max_pool_3d(Problem):
    """3D max pooling problem."""

    is_exact = False

    parameters = [
        {"name": "input", "type": "float", "pointer": True, "const": True},
        {"name": "kernel_size", "type": "int", "pointer": False, "const": False},
        {"name": "stride", "type": "int", "pointer": False, "const": False},
        {"name": "padding", "type": "int", "pointer": False, "const": False},
        {"name": "dilation", "type": "int", "pointer": False, "const": False},
        {"name": "output", "type": "float", "pointer": True, "const": False},
        {"name": "H", "type": "size_t", "pointer": False, "const": False},
        {"name": "W", "type": "size_t", "pointer": False, "const": False},
        {"name": "D", "type": "size_t", "pointer": False, "const": False},
    ]

    def __init__(self):
        super().__init__(
            name="max-pool-3d"
        )

    def reference_solution(self, input_tensor: torch.Tensor, kernel_size: int, 
                         stride: int, padding: int, dilation: int) -> torch.Tensor:
        """
        PyTorch implementation of 3D max pooling.

        Args:
            input_tensor: Input tensor of shape (H, W, D)
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window
            padding: Padding to be applied before pooling
            dilation: Spacing between kernel elements

        Returns:
            Result of max pooling
        """
        with torch.no_grad(), torch.autocast("cuda", enabled=False, dtype=input_tensor.dtype):
            input_reshaped = input_tensor.view(1, 1, input_tensor.size(0), input_tensor.size(1), input_tensor.size(2))

            result = torch.nn.functional.max_pool3d(
                input_reshaped,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )

            return result.view(result.size(2), result.size(3), result.size(4))

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """
        Generate test cases for 3D max pooling.

        Returns:
            List of test case dictionaries with varying sizes
        """
        dtype = self.param_dtype(0)

        test_configs = [
            (256, 256, 256, 2, 2, 0, 2),
            (512, 512, 512, 3, 2, 1, 1),
            (1024, 1024, 1024, 4, 4, 2, 1),
            (512, 512, 512, 3, 3, 1, 3),
            (1024, 1024, 1024, 5, 2, 2, 2),
            (1024, 1024, 1024, 7, 3, 3, 1)
        ]

        test_cases = []
        for h, w, d, k, s, p, dilation in test_configs:
            seed = Problem.get_seed(f"{self.name}_H={h}_W={w}_D={d}_K={k}_S={s}_P={p}_dil={dilation}")
            test_cases.append({
                "name": f"H={h}, W={w}, D={d}, K={k}, S={s}, P={p}, dilation={dilation}",
                "height": h,
                "width": w,
                "depth": d,
                "kernel_size": k,
                "stride": s,
                "padding": p,
                "dilation": dilation,
                "create_inputs": lambda h=h, w=w, d=d, k=k, s=s, p=p, dilation=dilation, seed=seed, dtype=dtype: (
                    *(lambda g: (
                        torch.rand((h, w, d), device="cuda", dtype=dtype, generator=g) * 2.0 - 1.0,  # uniform [-1, 1]
                    ))(torch.Generator(device="cuda").manual_seed(seed)),
                    k, s, p, dilation
                )
            })
        return test_cases

    def generate_sample(self) -> List[Dict[str, Any]]:
        """
        Generate a single sample test case for 3D max pooling with predictable input.

        Returns:
            A list containing a single test case dictionary
        """
        dtype = self.param_dtype(0)

        return {
            "name": "sample_basic_2x2x2",
            "height": 4,
            "width": 4,
            "depth": 4,
            "kernel_size": 2,
            "stride": 1,
            "padding": 0,
            "dilation": 1,
            "create_inputs": lambda h=4, w=4, d=4, k=2, s=1, p=0, dil=1: (
                torch.arange(h * w * d, device="cuda", dtype=dtype).reshape(h, w, d) / (h * w * d),
                k,
                s,
                p,
                dil
            )
        }

    def verify_result(self, expected_output: torch.Tensor, 
                     actual_output: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if the max pooling result is correct.

        Args:
            expected_output: Output from reference solution
            actual_output: Output from submitted solution

        Returns:
            Tuple of (is_correct, debug_info)
        """
        is_close = torch.allclose(actual_output, expected_output, rtol=9e-5, atol=9e-5)

        debug_info = {}
        if not is_close:
            diff = actual_output - expected_output
            max_diff = torch.max(torch.abs(diff)).item()
            mean_diff = torch.mean(torch.abs(diff)).item()

            # Find indices of largest differences
            flat_diff = diff.flatten()
            _, top_indices = torch.topk(torch.abs(flat_diff), min(5, flat_diff.numel()))

            # Convert flat indices back to 3D coordinates
            h, w, d = expected_output.shape
            sample_diffs = {}
            for i, idx in enumerate(top_indices):
                row = idx.item() // (w * d)
                col = (idx.item() // d) % w
                depth = idx.item() % d
                sample_diffs[f"({row}, {col}, {depth})"] = {
                    "expected": expected_output[row, col, depth].item(),
                    "actual": actual_output[row, col, depth].item(),
                    "diff": diff[row, col, depth].item()
                }

            debug_info = {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
                "sample_differences": sample_diffs
            }

        return is_close, debug_info

    def get_flops(self, test_case: Dict[str, Any]) -> int:
        """
        Get the number of floating point operations for the problem.

        Args:
            test_case: The test case dictionary

        IMPORTANT: For max pooling, we count comparisons as FLOPs.
        Each output element requires (kernel_size * kernel_size * kernel_size - 1) comparisons.

        Returns:
            Number of floating point operations
        """
        H = test_case["height"]
        W = test_case["width"]
        D = test_case["depth"]
        K = test_case["kernel_size"]
        S = test_case["stride"]
        P = test_case["padding"]

        # Calculate output dimensions
        H_out = ((H + 2 * P - K) // S) + 1
        W_out = ((W + 2 * P - K) // S) + 1
        D_out = ((D + 2 * P - K) // S) + 1
        # Each output element requires K*K*K-1 comparisons
        comparisons_per_output = K * K * K - 1

        # Total FLOPs (comparisons) for the entire output
        return H_out * W_out * D_out * comparisons_per_output

    def get_extra_params(self, test_case: Dict[str, Any]) -> List[Any]:
        """
        Get extra parameters to pass to the CUDA solution.

        Args:
            test_case: The test case dictionary

        Returns:
            List containing the image height H, width W, depth D, kernel_size, stride, padding, and dilation
        """
        return [
            test_case["height"],
            test_case["width"],
            test_case["depth"]
        ]
