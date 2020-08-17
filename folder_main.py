# main file to process the whole folder

import sys
from os import listdir
from os.path import isfile, join
from DSL import run_DSL, get_list_diff
from input_utils import parse_api_signature, list_all_differences, apply_transformation

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('usage: main.py old_api_signature new_api_signature folder_path')
        sys.exit(1)

    old_signature_string = sys.argv[1]
    new_signature_string = sys.argv[2]
    folder_path = sys.argv[3]
    old_signature = parse_api_signature(old_signature_string)
    new_signature = parse_api_signature(new_signature_string)
    dsl_list = list_all_differences(old_signature, new_signature)
    # get list of files
    list_file = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
    # list_path = []
    # for file in list_file:
    #     list_path.append(join(folder_path, f))
    for file in list_file:
        print("Processing file: " + file)
        modified_tree, list_change = run_DSL(dsl_list, file, old_signature)
        file_change_dictionary = get_list_diff(modified_tree, list_change, file)
        print("File change dictionary")
        print(file_change_dictionary)
        apply_transformation(file_change_dictionary, file)
        print()
        print()