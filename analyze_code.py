import ast
import os
import re

# 1. Define target files and data structures
TARGET_FILES = [
    "python/VAE.py",
    "python/VAE_prob.py",
    "python/enc_dec.py",
    "python/hidden_layers.py",
    "python/losses.py",
    "python/rep_trick.py",
    "python/train_VAE_prob.py",
    "python/train_VAE_prob_3.py",
]

# Data Structures
files = set()
classes = []  # List of dicts: {name, file, bases, methods}
functions = []  # List of dicts: {name, file}
imports = []  # List of dicts: {from_file, to_module, names, type}
calls = []  # List of dicts: {caller_file, caller_context, callee_name, type, object_name (for method)}
instantiations = []  # List of dicts: {file, context, class_name}

# --- Helper functions for DOT ID generation ---
def clean_name_for_dot(name):
    """Cleans a name to be a valid DOT identifier."""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not name or not name[0].isalpha() and name[0] != '_':
        name = "_" + name # Ensure it starts with a letter or underscore
    return name

def file_to_dot_id(filepath):
    """Generates a DOT ID for a file path."""
    return "file_" + clean_name_for_dot(filepath)

def class_to_dot_id(class_info_or_name, file_path=None):
    """Generates a DOT ID for a class."""
    if isinstance(class_info_or_name, dict):
        return "class_" + clean_name_for_dot(class_info_or_name['file']) + "_" + clean_name_for_dot(class_info_or_name['name'])
    elif file_path and isinstance(class_info_or_name, str):
        return "class_" + clean_name_for_dot(file_path) + "_" + clean_name_for_dot(class_info_or_name)
    else: # Fallback for just class name string (less unique)
         return "class_" + clean_name_for_dot(class_info_or_name)


def function_to_dot_id(func_info_or_name, file_path=None):
    """Generates a DOT ID for a top-level function."""
    if isinstance(func_info_or_name, dict):
        return "func_" + clean_name_for_dot(func_info_or_name['file']) + "_" + clean_name_for_dot(func_info_or_name['name'])
    elif file_path and isinstance(func_info_or_name, str):
         return "func_" + clean_name_for_dot(file_path) + "_" + clean_name_for_dot(func_info_or_name)
    else: # Fallback for just function name string
        return "func_" + clean_name_for_dot(func_info_or_name)


def method_to_dot_id(class_name, method_name, file_path):
    """Generates a DOT ID for a method."""
    return "method_" + clean_name_for_dot(file_path) + "_" + clean_name_for_dot(class_name) + "_" + clean_name_for_dot(method_name)

def context_to_dot_id(file_path, context_str):
    """
    Generates a DOT ID from a context string (e.g., "ClassName.method_name", "function_name").
    """
    if not context_str or context_str == "global" or context_str == "<class_scope>": # Calls from global or class scope
        return file_to_dot_id(file_path) # Source is the file itself

    parts = context_str.split('.')
    if len(parts) == 2: # ClassName.method_name
        return method_to_dot_id(parts[0], parts[1], file_path)
    else: # function_name
        return function_to_dot_id(parts[0], file_path)


# Helper to get all defined class names (used for call analysis)
def get_defined_class_names_map(): # Returns {name: file_path}
    return {cls['name']: cls['file'] for cls in classes}

# Helper to get all defined top-level function names (used for call analysis)
def get_defined_function_names_map(): # Returns {name: file_path}
    return {fn['name']: fn['file'] for fn in functions}


def get_enclosing_context(node_to_find, tree, filepath):
    path = []
    def find_path(current_node):
        path.append(current_node)
        if current_node == node_to_find: return True
        for child in ast.iter_child_nodes(current_node):
            if find_path(child): return True
        path.pop()
        return False

    if not find_path(tree): return "unknown"

    enclosing_func_def, enclosing_class_def = None, None
    for p_node in reversed(path[:-1]):
        if isinstance(p_node, ast.FunctionDef):
            enclosing_func_def = p_node; break
        if isinstance(p_node, ast.ClassDef):
            enclosing_class_def = p_node; break
    
    if enclosing_func_def:
        func_def_index_in_path = path.index(enclosing_func_def)
        parent_of_func_def = None
        for i in range(func_def_index_in_path - 1, -1, -1):
            path_node = path[i]
            if isinstance(path_node, ast.ClassDef):
                if any(body_item == enclosing_func_def for body_item in path_node.body):
                    parent_of_func_def = path_node; break
        if parent_of_func_def: return f"{parent_of_func_def.name}.{enclosing_func_def.name}"
        else: return enclosing_func_def.name
    elif enclosing_class_def: return f"{enclosing_class_def.name}.<class_scope>"
    return "global"


def analyze_file_content(filepath, content):
    try:
        tree = ast.parse(content, filename=filepath)
    except SyntaxError as e:
        print(f"Error parsing AST for {filepath}: {e}"); return
    
    current_file_dir = os.path.dirname(filepath)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
            base_names = []
            for base in node.bases:
                if isinstance(base, ast.Name): base_names.append(base.id)
                elif isinstance(base, ast.Attribute):
                    attr_path = []
                    curr_attr = base
                    while isinstance(curr_attr, ast.Attribute):
                        attr_path.append(curr_attr.attr)
                        curr_attr = curr_attr.value
                    if isinstance(curr_attr, ast.Name): attr_path.append(curr_attr.id)
                    base_names.append(".".join(reversed(attr_path)))
                else: base_names.append(f"<complex_base:{type(base).__name__}>")
            classes.append({"name": node.name, "file": filepath, "bases": base_names, "methods": method_names})
        elif isinstance(node, ast.FunctionDef):
            if any(module_child_node == node for module_child_node in tree.body):
                functions.append({"name": node.name, "file": filepath})

    defined_class_map = get_defined_class_names_map()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module if node.module else ""
            resolved_module_name = ""
            if node.level > 0: resolved_module_name = "." * node.level + module_name
            elif module_name:
                potential_module_file = f"python/{module_name.split('.')[0]}.py"
                resolved_module_name = module_name.split('.')[0] if potential_module_file in TARGET_FILES else module_name
            
            imported_names = [alias.name for alias in node.names]
            final_to_module = resolved_module_name.lstrip('.')
            if not node.module and node.level > 0 : # from . import foo
                 final_to_module = os.path.basename(current_file_dir) # 'python'

            imports.append({
                "from_file": filepath, "to_module": final_to_module, 
                "names": imported_names, "type": "ImportFrom"
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                imports.append({
                    "from_file": filepath, "to_module": module_name.split('.')[0], 
                    "names": [alias.asname if alias.asname else alias.name], "type": "Import"
                })
        elif isinstance(node, ast.Call):
            context = get_enclosing_context(node, tree, filepath)
            callee_name_str, object_name_str = "", None

            if isinstance(node.func, ast.Name):
                callee_name_str = node.func.id
                if callee_name_str in defined_class_map:
                    instantiations.append({"file": filepath, "context": context, "class_name": callee_name_str})
                else:
                    calls.append({
                        "caller_file": filepath, "caller_context": context, "callee_name": callee_name_str,
                        "type": "function_call", "object_name": None
                    })
            elif isinstance(node.func, ast.Attribute):
                callee_name_str = node.func.attr
                if isinstance(node.func.value, ast.Name): object_name_str = node.func.value.id
                elif isinstance(node.func.value, ast.Attribute):
                    val = node.func.value; obj_parts = [val.attr]
                    while isinstance(val.value, ast.Attribute): val = val.value; obj_parts.append(val.attr)
                    if isinstance(val.value, ast.Name): obj_parts.append(val.value.id); object_name_str = ".".join(reversed(obj_parts))
                    else: object_name_str = "<complex_object>"
                elif isinstance(node.func.value, ast.Call): object_name_str = "<call_result>"
                
                calls.append({
                    "caller_file": filepath, "caller_context": context, "callee_name": callee_name_str,
                    "type": "method_call", "object_name": object_name_str
                })

def generate_dot_output(output_filename="python_code_relations.dot"):
    """Generates a DOT language representation of the code relationships."""
    dot_lines = []
    dot_lines.append("digraph CodeRelations {")
    dot_lines.append('  rankdir="LR";')
    dot_lines.append('  fontsize="10";')
    dot_lines.append('  node [shape=record, fontsize=10, style=filled];')
    dot_lines.append('  edge [fontsize=9];')
    dot_lines.append("")

    processed_files_for_dot = set() 
    external_modules_dot_ids = set()

    for file_path in files: 
        file_dot_id = file_to_dot_id(file_path)
        processed_files_for_dot.add(file_dot_id)
        dot_lines.append(f'  subgraph "cluster_{file_dot_id}" {{')
        dot_lines.append(f'    label="{os.path.basename(file_path)}";')
        dot_lines.append('    color=lightgrey;')
        dot_lines.append('    style=filled;')
        dot_lines.append("")

        for class_info in classes:
            if class_info['file'] == file_path:
                class_id = class_to_dot_id(class_info)
                methods_label = "\\l".join([f"+ {method_name}()" for method_name in class_info['methods']])
                if not methods_label:
                     node_label = f"{{{class_info['name']}}}"
                else:
                    node_label = f"{{{class_info['name']} | {methods_label}\\l}}"
                dot_lines.append(f'    {class_id} [label="{node_label}", fillcolor=lightblue];')

        for func_info in functions:
            if func_info['file'] == file_path:
                func_id = function_to_dot_id(func_info)
                dot_lines.append(f'    {func_id} [label="{{Function: {func_info["name"]}()}}", fillcolor=lightgreen];')
        dot_lines.append("  }")
        dot_lines.append("")

    defined_class_ids = {class_to_dot_id(cls) for cls in classes}
    for class_info in classes:
        from_class_id = class_to_dot_id(class_info)
        for base_name in class_info['bases']:
            base_class_file = None
            for c in classes: 
                if c['name'] == base_name:
                    base_class_file = c['file']; break
            
            if base_class_file: 
                to_class_id = class_to_dot_id(base_name, base_class_file)
                if to_class_id in defined_class_ids:
                     dot_lines.append(f'  {from_class_id} -> {to_class_id} [label="inherits_from", style=dashed, arrowhead=empty];')

    target_file_basename_map = {os.path.basename(f).replace(".py", ""): file_to_dot_id(f) for f in TARGET_FILES}

    for imp_info in imports:
        from_dot_id = file_to_dot_id(imp_info['from_file'])
        to_module_cleaned = clean_name_for_dot(imp_info['to_module'])
        to_dot_id = None

        if imp_info['to_module'] in target_file_basename_map:
            to_dot_id = target_file_basename_map[imp_info['to_module']]
        elif imp_info['to_module'].startswith("python"):
             to_dot_id = file_to_dot_id(imp_info['from_file']) 
        else: 
            ext_mod_id = "extmodule_" + to_module_cleaned
            if ext_mod_id not in external_modules_dot_ids:
                dot_lines.append(f'  {ext_mod_id} [label="{imp_info["to_module"]}", fillcolor=ivory, shape=box3d];')
                external_modules_dot_ids.add(ext_mod_id)
            to_dot_id = ext_mod_id
        
        if to_dot_id:
            import_names_label = ", ".join(imp_info['names'])
            dot_lines.append(f'  {from_dot_id} -> {to_dot_id} [label="imports: {import_names_label}"];')

    for inst_info in instantiations:
        source_id = context_to_dot_id(inst_info['file'], inst_info['context'])
        target_class_file = None
        for c in classes: 
            if c['name'] == inst_info['class_name']:
                target_class_file = c['file']; break
        if target_class_file:
            target_id = class_to_dot_id(inst_info['class_name'], target_class_file)
            if target_id in defined_class_ids: 
                 dot_lines.append(f'  {source_id} -> {target_id} [label="instantiates", color=purple, arrowhead=vee];')

    all_known_methods_map = {} 
    for c in classes:
        for m_name in c['methods']:
            m_id = method_to_dot_id(c['name'], m_name, c['file'])
            all_known_methods_map[(c['file'], c['name'], m_name)] = m_id
    
    all_known_functions_map = {}
    for f in functions:
        f_id = function_to_dot_id(f)
        all_known_functions_map[(f['file'], f['name'])] = f_id

    for call_info in calls:
        source_id = context_to_dot_id(call_info['caller_file'], call_info['caller_context'])
        target_id = None

        if call_info['type'] == 'function_call':
            found_func = None
            for f_file, f_name in all_known_functions_map.keys():
                if f_name == call_info['callee_name']:
                    found_func = all_known_functions_map[(f_file, f_name)]; break 
            if found_func: target_id = found_func
        elif call_info['type'] == 'method_call':
            object_name = call_info['object_name']
            callee_method_name = call_info['callee_name']
            target_class_name = None
            target_class_file = call_info['caller_file']
            if object_name == 'self':
                if '.' in call_info['caller_context']:
                    target_class_name = call_info['caller_context'].split('.')[0]
            if target_class_name:
                if (target_class_file, target_class_name, callee_method_name) in all_known_methods_map:
                    target_id = all_known_methods_map[(target_class_file, target_class_name, callee_method_name)]
            else: 
                for f, c_name, m_name in all_known_methods_map.keys():
                    if m_name == callee_method_name:
                        target_id = all_known_methods_map[(f, c_name, m_name)]; break 

        if source_id and target_id:
            if source_id == target_id and source_id.startswith("file_"): continue
            dot_lines.append(f'  {source_id} -> {target_id} [label="calls", color=darkgreen, arrowhead=open];')
        elif source_id and not target_id and call_info['type'] == 'method_call' and call_info['object_name']:
            unresolved_target_id = f"unresolved_method_{clean_name_for_dot(call_info['object_name'])}_{clean_name_for_dot(call_info['callee_name'])}"
            if unresolved_target_id not in external_modules_dot_ids: 
                 dot_lines.append(f'  {unresolved_target_id} [label="{call_info["object_name"]}.{call_info["callee_name"]}()", fillcolor=gold, shape=box];')
                 external_modules_dot_ids.add(unresolved_target_id)
            dot_lines.append(f'  {source_id} -> {unresolved_target_id} [label="calls?", color=orange, style=dotted, arrowhead=open];')

    dot_lines.append("}")
    dot_lines.append("") 

    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(dot_lines))
        print(f"DOT file '{output_filename}' generated successfully.")
        # Return success for main function to print further instructions
        return True 
    except IOError as e:
        print(f"Error writing DOT file {output_filename}: {e}")
        return False


def main():
    """
    Main function to drive the analysis of specified Python files.
    """
    dot_output_filename = "python_code_relations.dot" # Define filename here for instructions

    for py_file in TARGET_FILES:
        files.add(py_file)
        try:
            with open(py_file, "r", encoding="utf-8") as source_file:
                content = source_file.read()
                analyze_file_content(py_file, content)
        except FileNotFoundError:
            print(f"Error: File not found - {py_file}"); continue
        except Exception as e:
            print(f"Error reading file {py_file}: {e}"); continue
    
    all_method_signatures = set()
    for cls_info in classes:
        for method_name in cls_info["methods"]:
            all_method_signatures.add((cls_info["file"], method_name))
    
    global functions
    functions = [f_info for f_info in functions if (f_info["file"], f_info["name"]) not in all_method_signatures]

    print("\nAnalysis complete. Data collected.")
    print(f"Processed Files: {len(files)}")
    print(f"Classes Defined: {len(classes)}")
    print(f"Top-level Functions Defined: {len(functions)}")
    print(f"Imports Found: {len(imports)}")
    print(f"Function/Method Calls Found: {len(calls)}")
    print(f"Class Instantiations Found: {len(instantiations)}")

    if generate_dot_output(dot_output_filename): # Call DOT generation
        # Append visualization instructions
        print("\n---------------------------------------------------------------------")
        print("To visualize the relationships, you need Graphviz installed.")
        print("If you have Graphviz, you can generate an image from the .dot file using:\n")
        print(f"For a PNG image:")
        print(f"    dot -Tpng {dot_output_filename} -o {dot_output_filename.replace('.dot', '.png')}\n")
        print(f"For an SVG (vector) image:")
        print(f"    dot -Tsvg {dot_output_filename} -o {dot_output_filename.replace('.dot', '.svg')}\n")
        print("This will create an image file (e.g., python_code_relations.png)")
        print("in the same directory, showing the code relationship map.")
        print("---------------------------------------------------------------------")


if __name__ == "__main__":
    main()
