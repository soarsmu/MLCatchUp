# main file to process the whole folder

import sys
from os import listdir
from os.path import isfile, join
from DSL import run_DSL, get_list_diff
from input_utils import parse_api_signature, list_all_differences, apply_transformation

if __name__ == "__main__":
    # Parsing arguments
    argument_valid = True
    isInferPhase = True
    if len(sys.argv) < 6:
        argument_valid = False
    else:
        if sys.argv[1].lower() == "--infer":
            if len(sys.argv) != 6:
                argument_valid = False
            else:
                if sys.argv[4].lower() != "--output":
                    argument_valid = False
        elif sys.argv[1].lower() == "--transform":
            isInferPhase = False
            if len(sys.argv) != 8:
                argument_valid = False
            else:
                if sys.argv[2].lower() != "--dsl":
                    argument_valid = False
                if sys.argv[4].lower() != "--input":
                    argument_valid = False
                if sys.argv[6].lower() != "--output":
                    argument_valid = False
    if not argument_valid:
        print("Parsed arguments invalid:")
        print("Usage:")
        print("Inference: main.py --infer <deprecated_api_signature> <updated_api_signature> --output <dsl_script_filepath>")
        print("Update: main.py -- transform --dsl <dsl_script_filepath> --input <deprecated_filepath> --output <output_filepath>")
        print("Exiting...")
        sys.exit(1)

    if isInferPhase:
        old_signature_string = sys.argv[2]
        new_signature_string = sys.argv[3]
        output_path = sys.argv[5]
        print("Transformation Inference Process ")
        print("    Deprecated API : " + old_signature_string)
        print("    Updated API    : " + new_signature_string)
        # transformation inference process
        old_signature = parse_api_signature(old_signature_string)
        new_signature = parse_api_signature(new_signature_string)
        dsl_list = list_all_differences(old_signature, new_signature)
        print("Inferred DSL: ")
        for dsl in dsl_list:
            print("    - " + dsl)
        with open(output_path, "w", encoding="utf-8") as f:
            for dsl in dsl_list:
                f.write(dsl)
                f.write("\n")
            f.close()
    else:
        # update application process
        dsl_filepath = sys.argv[3]
        input_filepath = sys.argv[5]
        output_path = sys.argv[7]
        print("Transformation Application Process ")
        print("    DSL Scripts:")
        with open(dsl_filepath, "r", encoding="utf-8") as dsl_file:
            dsl_scripts = dsl_file.readlines()
            for dsl in dsl_scripts:
                print("        - " + dsl)
        dsl_file.close()
        # Get the old API signature from the DSL
        splitted_dsl_string = dsl_scripts[0].split(" ")
        old_signature_string = ""
        if splitted_dsl_string[0] == "rename_method":
            old_signature_string = splitted_dsl_string[1]
        elif splitted_dsl_string[0] == "rename_parameter":
            old_signature_string = splitted_dsl_string[5]
        elif splitted_dsl_string[0] == "remove_parameter":
            old_signature_string = splitted_dsl_string[3]
        elif splitted_dsl_string[0] == "positional_to_keyword":
            old_signature_string = splitted_dsl_string[6]
        elif splitted_dsl_string[0] == "add_parameter":
            old_signature_string = splitted_dsl_string[5]
        elif splitted_dsl_string[0] == "remove_api":
            old_signature_string = splitted_dsl_string[1]

        old_signature = parse_api_signature(old_signature_string)

        modified_tree, list_change, hasConstraint, code_string, parameter_name = run_DSL(dsl_scripts, input_filepath, old_signature)
        file_change_dictionary = get_list_diff(modified_tree, list_change, input_filepath)
        print("    List of Changes: ")
        for key, item in file_change_dictionary.items():
            print("        Line " + key.__str__() + ":")
            print("            original: " + item[0])
            print("            updated : " + "".join(item[1]))
        apply_transformation(file_change_dictionary, input_filepath, hasConstraint, code_string, parameter_name, output_path)
        print("File updated successfully!!!")