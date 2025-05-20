import os
import glob
import sys
import importlib.util
import inspect
import torch
import numpy as np

# Get the absolute path to the project root directory (where baselines.py is)
project_root = os.path.abspath(os.path.dirname(__file__))

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from problem import Problem # Assuming problem.py is in the project root
except ImportError as e:
    print(f"CRITICAL: Failed to import 'Problem' class from 'problem.py'. Ensure 'problem.py' exists in project root: {project_root}")
    print(f"Error details: {e}")
    sys.exit(1)

PROBLEMS_DIR = "problems"

def verify_problem_solutions(problem_dir_path):
    problem_name = os.path.basename(problem_dir_path)
    def_file_path = os.path.join(problem_dir_path, "def.py")
    solution_file_path = os.path.join(problem_dir_path, "solution.py")

    print(f"\nVerifying solutions for problem: {problem_name}")

    if not os.path.exists(def_file_path):
        print(f"  ERROR: def.py not found at {def_file_path}. Skipping {problem_name}.")
        return
    if not os.path.exists(solution_file_path):
        print(f"  ERROR: solution.py not found at {solution_file_path}. This is required by def.py. Skipping {problem_name}.")
        return

    problem_instance = None
    problem_class_loaded = False
    def_module_name = f"problems.{problem_name}.def"
    tinygrad_module_name = f"problems.{problem_name}.solution" # Define here for cleanup scope

    try:
        spec = importlib.util.spec_from_file_location(def_module_name, def_file_path)
        if spec is None or spec.loader is None:
            print(f"  ERROR: Could not create module spec for {def_module_name} from {def_file_path}")
            return

        module = importlib.util.module_from_spec(spec)
        module.__package__ = f"problems.{problem_name}"
        
        sys.modules[def_module_name] = module
        print(f"  DEBUG: Executing module {def_module_name} from {def_file_path} with package {module.__package__}")
        spec.loader.exec_module(module)
        print(f"  DEBUG: Finished executing module {def_module_name}")

        problem_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Problem) and obj is not Problem:
                problem_class = obj
                break
        
        if problem_class is None:
            print(f"  ERROR: Could not find a class inheriting from Problem in {def_file_path}")
            return

        print(f"  DEBUG: Found problem_class: {problem_class.__name__} in module {module}")
        print(f"  DEBUG: MRO for {problem_class.__name__}: {problem_class.mro()}")

        # Check for reference_solution specifically on the class BEFORE instantiation
        if hasattr(problem_class, 'reference_solution'):
            method_obj = problem_class.reference_solution
            print(f"  DEBUG: 'reference_solution' found in problem_class attributes. Type: {type(method_obj)}")
            # Check if it's an abstract method directly on the class (won't show if overridden by concrete)
            is_abstract = getattr(method_obj, '__isabstractmethod__', False)
            print(f"  DEBUG: Is problem_class.reference_solution marked abstract? {is_abstract}")
            if not is_abstract and callable(method_obj):
                 # For a classmethod/staticmethod, check the underlying function if possible
                if isinstance(method_obj, (classmethod, staticmethod)):
                    is_abstract_on_func = getattr(method_obj.__func__, '__isabstractmethod__', False)
                    print(f"  DEBUG: Underlying function for {type(method_obj)} is abstract? {is_abstract_on_func}")
                print(f"  DEBUG: problem_class.reference_solution appears concrete and callable.")
            elif is_abstract:
                 print(f"  DEBUG: problem_class.reference_solution IS STILL ABSTRACT on the class object itself.")
            else:
                 print(f"  DEBUG: problem_class.reference_solution is present but not callable or has an issue.")

        else:
            print(f"  DEBUG: 'reference_solution' NOT FOUND in problem_class attributes/MRO directly via hasattr.")


        problem_instance = problem_class() # <<<< THIS IS WHERE THE ORIGINAL ERROR OCCURS
        problem_class_loaded = True
        print(f"  DEBUG: Successfully instantiated {problem_class.__name__}")


    except ImportError as e_exec:
        print(f"  ERROR: ImportError during loading of {def_module_name} (possibly from '.solution'): {e_exec}")
        return
    except Exception as e: # This catches the TypeError during problem_class()
        print(f"  ERROR: Error processing problem definition for {problem_name} from {def_file_path}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for this error
        return
    # finally:
        # Module cleanup is now at the end of the main verify_problem_solutions function

    if not problem_class_loaded or problem_instance is None:
        print(f"  Skipping {problem_name} due to previous errors in loading its definition.")
        return

    # --- Solution loading and verification ---
    tinygrad_solution_func = None
    try:
        # Try to get reference_tinygrad_solution from the instance first
        if hasattr(problem_instance, 'reference_tinygrad_solution') and callable(problem_instance.reference_tinygrad_solution):
            tinygrad_solution_func = problem_instance.reference_tinygrad_solution
            print(f"  DEBUG: Found 'reference_tinygrad_solution' method on problem_instance.")
        else:
            print(f"  DEBUG: 'reference_tinygrad_solution' not found as a callable method on problem_instance. Will try loading solution module separately.")
            # Fallback: Load solution.py module to find the function (less ideal if it's a method)
            spec_sol = importlib.util.spec_from_file_location(tinygrad_module_name, solution_file_path)
            if spec_sol is None or spec_sol.loader is None:
                print(f"  ERROR: Could not create module spec for {tinygrad_module_name} from {solution_file_path}")
            else:
                tinygrad_module = importlib.util.module_from_spec(spec_sol)
                tinygrad_module.__package__ = f"problems.{problem_name}"
                sys.modules[tinygrad_module_name] = tinygrad_module
                spec_sol.loader.exec_module(tinygrad_module)
                print(f"  DEBUG: Loaded solution module {tinygrad_module_name}")

                # Attempt to find a suitable function in the loaded solution module
                for name, obj in inspect.getmembers(tinygrad_module):
                    if inspect.isfunction(obj) and "tinygrad" in name.lower():
                        tinygrad_solution_func = obj
                        print(f"  DEBUG: Found potential tinygrad solution function in module: {name}")
                        break
                if tinygrad_solution_func is None: # Last resort for single function in module
                    defined_functions = [obj for name, obj in inspect.getmembers(tinygrad_module) if inspect.isfunction(obj) and inspect.getmodule(obj) is tinygrad_module]
                    if len(defined_functions) == 1:
                        tinygrad_solution_func = defined_functions[0]
                        print(f"  DEBUG: Using the only defined function in solution module: {tinygrad_solution_func.__name__}")


        if tinygrad_solution_func is None:
            print(f"  WARNING: Could not find tinygrad solution function for {problem_name}. Skipping tinygrad verification.")
        else:
            # --- Test case generation and verification ---
            dtypes = [torch.float32] # Using only float32 for brevity now, add torch.float64 if needed
            for dtype in dtypes:
                print(f"    Testing with dtype: {dtype}")
                try:
                    test_cases_params = problem_instance.generate_test_cases(dtype)
                except NotImplementedError:
                    print(f"      generate_test_cases not implemented for {problem_name}. Skipping.")
                    continue
                except Exception as e_test_gen:
                    print(f"      Error generating test cases for {problem_name} with dtype {dtype}: {e_test_gen}")
                    continue

                if not test_cases_params:
                    print(f"      No test cases generated for {problem_name} with dtype {dtype}.")
                    continue

                for i, case_params in enumerate(test_cases_params):
                    case_name = case_params.get("name", f"Unnamed Case {i+1}")
                    print(f"      Test Case: {case_name}")
                    try:
                        inputs_tuple = case_params["create_inputs"]()
                        
                        ref_inputs_torch = [val.clone().cpu() if isinstance(val, torch.Tensor) else val for val in inputs_tuple]
                        
                        # Prepare inputs for tinygrad - this needs to be specific
                        # Assuming tinygrad function expects torch tensors as input based on ArgmaxSolutions example
                        # If it expects tinygrad.Tensor, conversion is needed here.
                        tinygrad_inputs_torch = [val.clone().cpu() if isinstance(val, torch.Tensor) else val for val in inputs_tuple]

                        ref_output = problem_instance.reference_solution(*ref_inputs_torch)
                        tinygrad_output_val = tinygrad_solution_func(*tinygrad_inputs_torch)

                        ref_output_np = ref_output.detach().cpu().numpy() if isinstance(ref_output, torch.Tensor) else np.array(ref_output)
                        
                        if isinstance(tinygrad_output_val, torch.Tensor):
                            tinygrad_output_np = tinygrad_output_val.detach().cpu().numpy()
                        elif hasattr(tinygrad_output_val, 'numpy'): # For tinygrad.Tensor
                            tinygrad_output_np = tinygrad_output_val.numpy()
                        else:
                            tinygrad_output_np = np.array(tinygrad_output_val)

                        tolerance = {torch.float32: 1e-5, torch.float64: 1e-8}.get(dtype, 1e-5)
                        outputs_match = np.allclose(ref_output_np, tinygrad_output_np, atol=tolerance, rtol=tolerance)

                        result_str = "MATCH" if outputs_match else "DO NOT MATCH"
                        print(f"        Reference vs Tinygrad: {result_str}")
                        if not outputs_match:
                             print(f"          Max Diff: {np.max(np.abs(ref_output_np - tinygrad_output_np)) if ref_output_np.shape == tinygrad_output_np.shape else 'Shape Mismatch'}")


                    except Exception as e_verify:
                        print(f"        Error during verification of test case {case_name}: {e_verify}")
                        import traceback
                        traceback.print_exc()
    
    except Exception as e_outer_sol: # Catch errors related to solution module loading if any part fails
        print(f"  ERROR: Outer error related to solution processing for {problem_name}: {e_outer_sol}")
        import traceback
        traceback.print_exc()


    # Clean up modules from sys.modules to prevent side-effects for subsequent problem loadings
    if def_module_name in sys.modules:
        del sys.modules[def_module_name]
        print(f"  DEBUG: Unloaded module {def_module_name}")
    if tinygrad_module_name in sys.modules:
        del sys.modules[tinygrad_module_name]
        print(f"  DEBUG: Unloaded module {tinygrad_module_name}")


def main():
    abs_problems_dir = os.path.join(project_root, PROBLEMS_DIR)
    problem_dirs_glob = glob.glob(os.path.join(abs_problems_dir, "*"))

    if not problem_dirs_glob:
        print(f"No problem directories found in {abs_problems_dir}. Please check the PROBLEMS_DIR path and directory structure.")
        return

    # Filter out files, keep only directories
    problem_dirs_list = sorted([p for p in problem_dirs_glob if os.path.isdir(p)])
    
    if not problem_dirs_list:
        print(f"No actual problem sub-directories found in {abs_problems_dir}.")
        return

    for problem_dir_path_item in problem_dirs_list:
        verify_problem_solutions(problem_dir_path_item)

if __name__ == "__main__":
    print(f"Python sys.path[0]: {sys.path[0]}")
    print(f"Project root used for imports: {project_root}")
    main()