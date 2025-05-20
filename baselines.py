# This script will generate Tinygrad baseline solutions for all problems.

import os
import glob
import sys
import os
import ast
import inspect
import textwrap

# Add the root directory to sys.path to allow importing 'problem'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from problem import Problem

PROBLEMS_DIR = "problems"
SOLUTION_FILENAME = "solution.py"

def generate_tinygrad_solution(problem_dir):
    """Generates the Tinygrad solution for a single problem."""
    problem_name = os.path.basename(problem_dir)
    def_file_path = os.path.join(problem_dir, "def.py")
    problem_md_path = os.path.join(problem_dir, "problem.md")
    solution_file_path = os.path.join(problem_dir, SOLUTION_FILENAME)

    print(f"Generating Tinygrad solution for problem: {problem_name}")

    # Load problem definition module
    try:
        import importlib.util # Explicitly import here
        spec = importlib.util.spec_from_file_location("problem_definition", def_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"problem_definition_{problem_name}"] = module # Use unique name to avoid conflicts
        spec.loader.exec_module(module)

        # Find the problem class dynamically
        problem_class = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, Problem) and obj is not Problem:
                problem_class = obj
                break

        if problem_class is None:
            print(f"Error loading problem definition for {problem_name}: Could not find a class inheriting from Problem in {def_file_path}")
            return

        problem_instance = problem_class()
        reference_solution = problem_instance.reference_solution
        # We might need to read problem_md_path for input/output specs, but let's start with def.py
    except Exception as e:
        print(f"Error loading problem definition for {problem_name}: {e}")
        # If we can't load the problem definition, we can't generate a solution.
        return

    # Get the source code of the reference solution
    try:
        source_code = inspect.getsource(reference_solution)
    except Exception as e:
        print(f"Error getting source code for {problem_name}: {e}")
        return

    # Translate PyTorch to Tinygrad
    tinygrad_code = translate_pytorch_to_tinygrad(source_code, problem_name)

    if tinygrad_code is None:
        print(f"Could not translate PyTorch solution for {problem_name}")
        return


    # Write the Tinygrad solution to file
    try:
        with open(solution_file_path, "w") as f:
            f.write(tinygrad_code)
        print(f"Generated {solution_file_path}")
        # print(f"Generated content:\n{tinygrad_code}") # Optional: print generated content
    except Exception as e:
        print(f"Error writing solution file for {problem_name}: {e}")


def translate_pytorch_to_tinygrad(pytorch_source_code: str, problem_name: str) -> str | None:
    """
    Translates PyTorch source code to Tinygrad source code.

    Args:
        pytorch_source_code: The source code of the PyTorch reference solution.
        problem_name: The name of the problem.

    Returns:
        The generated Tinygrad source code as a string, or None if translation fails.
    """
    try:
        # Dedent the source code before parsing
        dedented_source_code = textwrap.dedent(pytorch_source_code)
        # Parse the AST
        tree = ast.parse(dedented_source_code)

        # Find the function definition node
        function_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_node = node
                break

        if function_node is None:
            print("Could not find function definition in PyTorch source code.")
            return None

        # Basic AST traversal and translation
        tinygrad_body_lines = []
        translated = False
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call):
                # Handle torch.argmax
                if isinstance(node.func, ast.Attribute) and \
                   isinstance(node.func.value, ast.Name) and \
                   node.func.value.id == 'torch' and node.func.attr == 'argmax':

                    if len(node.args) >= 2:
                        input_tensor_node = node.args[0]
                        dim_node = node.args[1]

                        # Assuming the input tensor is the first argument to the generated function
                        input_tensor_name = "args[0]" # This needs to be more robust

                        # Extract the dimension argument
                        dim_arg = ast.unparse(dim_node)

                        tinygrad_body_lines.append(f"input_tensor_tg = {input_tensor_name}")
                        tinygrad_body_lines.append(f"dim = {dim_arg}") # Assuming dim is passed as an argument
                        tinygrad_body_lines.append(f"return input_tensor_tg.argmax(axis=dim)")
                        translated = True
                        break # Assuming only one main operation for now

        # Construct the Tinygrad function string
        header = f"""# Tinygrad baseline solution for {problem_name}

from tinygrad.tensor import Tensor

def {problem_name}_tinygrad(*args):
    # This is a generated Tinygrad baseline solution.
    # The implementation is based on the PyTorch reference solution.

    # The reference solution can be accessed via the 'reference_solution' variable
    # if the problem definition was loaded successfully.
"""

        if translated:
            # Indent the body lines
            indented_body = textwrap.indent("\n".join(tinygrad_body_lines), "    ")
            tinygrad_code = header + "\n" + indented_body + "\n"
        else:
             # Placeholder for untranslated problems
             placeholder_body = f"""
    # TODO: Implement the translation from the PyTorch reference solution to Tinygrad.
    # The reference solution can be accessed via the 'reference_solution' variable
    # if the problem definition was loaded successfully.

    raise NotImplementedError(f"Tinygrad solution not yet automatically translated for {{problem_name}}")
"""
             indented_placeholder_body = textwrap.indent(placeholder_body, "    ")
             tinygrad_code = header + "\n" + indented_placeholder_body + "\n"


        return tinygrad_code

    except Exception as e:
        print(f"Error during PyTorch to Tinygrad translation for {problem_name}: {e}")
        return None


def main():
    """Main function to iterate through problems and generate solutions."""
    problem_dirs = glob.glob(os.path.join(PROBLEMS_DIR, "*"))

    for problem_dir in problem_dirs:
        if os.path.isdir(problem_dir):
            generate_tinygrad_solution(problem_dir)

if __name__ == "__main__":
    main()