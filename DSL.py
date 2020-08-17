import ast
import sys
import os
from ast_transform_rule import *

# Tutorial based on this article:
# https://dbader.org/blog/writing-a-dsl-with-python

# 1. Change parameter name
# 2. Change positional parameter into keyword parameter
# 3. Remove parameter
# 4. Change method name
# 5. Complex change


# 1:
from input_utils import ApiSignature

"change_param_name fully_qualified_api_name filename old_param_name new_param_name "

# 2:
"positional_to_keyword fully_qualified_api_name filename parameter_position keyword_param_name "

# 3:
"remove_param fully_qualified_api_name filename keyword_param_name "
"remove_param fully_qualified_api_name filename parameter_position "

# 4:
"change_method_name old_fully_qualified_name filename new_fully_qualified_name "

# 5:
"complex_transform api_name filename example_filename"

# dsl1.py


# Input: List of DSL commands, filename to be processed with the commands, and api_name
# Output: Tree, List Edited Lines
def run_DSL(list_DSL, filename, api_signature: ApiSignature):
    print("DSL COMMANDS: ")
    original_api_name = api_signature.api_name
    for DSL in list_DSL:
        print(DSL)
    with open(filename, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read())
        api_name = api_signature.api_name
        list_edited_line = {}
        for dsl in list_DSL:
            list_completed_api = get_list_API(tree, api_signature)
            list_line_number = get_list_line_number(list_completed_api)
            print("LIST COMPLETED API: " + list_completed_api.__str__())
            splitted_dsl = dsl.split(" ")
            print("LIST EDITED LINE: ")
            print(list_edited_line.__str__())
            if splitted_dsl[0] == "RENAME_API":
                print("RENAMING API")
                new_name = splitted_dsl[3]
                nameTransformer = ApiNameTransformer(api_name, new_name, list_line_number, list_completed_api)
                dict_change = nameTransformer.transform(tree)

                for key, value in dict_change.items():
                    list_edited_line[key] = value
                # Convert the API name into the new name for future detection
                api_name = new_name
                api_signature.api_name = new_name
            elif splitted_dsl[0] == "ADD_PARAM":
                print("ADDING PARAM")
                param_name = splitted_dsl[1]
                param_value = splitted_dsl[3]
                # currently default to call until there is new way to detect the type
                param_type = "call"
                addParamTransformer = AddNewParameter(api_name, param_name, param_type, param_value, list_line_number)
                dict_change = addParamTransformer.transform(tree)
                for key, value in dict_change.items():
                    list_edited_line[key] = value
            elif splitted_dsl[0] == "POSITIONAL_TO_KEYWORD":
                print("POSITIONAL TO KEYWORD")
                param_position = splitted_dsl[2]
                param_keyword = splitted_dsl[4]
                positionalToKeywordTransformer = PositionalToKeyword(api_name, param_position, param_keyword, list_line_number)
                dict_change = positionalToKeywordTransformer.transform(tree)
                for key, value in dict_change.items():
                    list_edited_line[key] = value
        print("Finished processing the DSL")
        print("List edited line: ")
        for key, value in list_edited_line.items():
            print("Line - " + key.__str__() + ": " + unparse(value))
        api_signature.api_name = original_api_name
        return tree, list_edited_line

# Input: modified tree and list diff from the Run_DSL function and the filename to be changed
# Output: Dictionary with key being the position of the diff and the value being a
# tuples of the old value and list of the new value
def get_list_diff(tree, list_diff, filename):
    list_position = list(list_diff.keys())
    list_position.remove(0)
    old_tree = ast.parse(open(filename, encoding="utf-8").read())

    with open("old_file.py", "w", encoding="utf-8") as old_open:
        old_open.write(unparse(old_tree))

    with open("new_file.py", "w", encoding="utf-8") as new_open:
        new_open.write(unparse(tree))

    old_list = unparse(old_tree).split("\n")
    new_list = unparse(tree).split("\n")

    diff_dict = {}
    i = 0
    position = 1
    while i < len(old_list):
        if old_list[i] == new_list[i]:
            old_list.pop(i)
            new_list.pop(i)
            position += 1
        elif new_list[i][0:4] == "from":
            print("FROM STATEMENT DETECTED")
            # Special case if import statement in the new key
            diff_dict[1] = "", [new_list.pop(i)]
            position += 1
        else:
            old_key = old_list.pop(i)
            new_key = [new_list.pop(i)]

            # while current_old != current_new:
            #     new_key.append(new_list.pop(i))
            #     current_new = new_list[i]
            #     print("DCGH")
            #     print(old_key)
            #     print(new_key)
            #     print("ABCD")
            #     print(current_new)
            #     print(current_old)

            # If cannot pop, probably an import statement, so just put it on front 1
            try:
                actual_position = list_position.pop(0)
            except:
                actual_position = 1
            while len(list_position) > 0 and list_position[0] < position:
                actual_position = list_position.pop(0)
            diff_dict[actual_position] = old_key, new_key
            position += 1
    return diff_dict

#
# # The source file is the 1st argument to the script
# if len(sys.argv) < 4:
#     print('usage: refer to help/readme. use DSL.py help for list of available commands')
#     sys.exit(1)
#
# DSL_MODE = sys.argv[1]
# ORIGINAL_API_NAME = sys.argv[2]
# FILENAME = sys.argv[3]
#
# if DSL_MODE == "change_param_name":
#     # Change param name
#     # Must have 4 additional arguments (sys.argv len must be 6)
#     if len(sys.argv) != 6:
#         print("usage: change_param_name fully_qualified_api_name filename old_param_name new_param_name")
#         sys.exit(1)
#     else:
#         print("Changing parameter name")
#         OLD_PARAM_NAME = sys.argv[4]
#         NEW_PARAM_NAME = sys.argv[5]
#
#
# if DSL_MODE == "positional_to_keyword":
#     # Positional parameter to keyword parameter
#     # Must have 4 additional arguments (sys.argv len must be 6)
#     if len(sys.argv) != 6:
#         print("usage: positional_to_keyword fully_qualified_api_name filename parameter_position keyword_param_name")
#         sys.exit(1)
#     else:
#         try:
#             PARAMETER_POSITION = int(sys.argv[4])
#         except Exception:
#             print("usage: positional_to_keyword fully_qualified_api_name filename parameter_position keyword_param_name")
#             sys.exit(1)
#         KEYWORD_PARAMETER_NAME = sys.argv[5]
#         print("Converting positional keyword")
#
#
# if DSL_MODE == "remove_param":
#     # Remove parameter
#     # Must have 3 additional arguments (sys.argv len must be 5)
#     if len(sys.argv) != 5:
#         print("usage: remove_param fully_qualified_api_name filename keyword_param_name")
#         print("usage: remove_param fully_qualified_api_name filename positional_param_position")
#         sys.exit(1)
#     else:
#         if sys.argv[4].isnumeric():
#             # removal of positional parameter
#             REMOVED_PARAM = int(sys.argv[4])
#         else:
#             REMOVED_PARAM = sys.argv[4]
#         print("Removing parameter")
#
# if DSL_MODE == "change_method_name":
#     # Change method name
#     # Must have 3 additional arguments (sys.argv len must be 5)
#     if len(sys.argv) != 5:
#         print("usage: change_method_name filename old_fully_qualified_name new_fully_qualified_name")
#         sys.exit(1)
#     else:
#         NEW_API_NAME = sys.argv[4]
#         print("Changing method name")
#
# if DSL_MODE == "complex_transform":
#     # Complex transformation
#     # Must have 3 additional arguments (sys.argv len must be 5)
#     if len(sys.argv) != 5:
#         print("usage: complex_transform fully_qualified_name filename example_filename")
#         sys.exit(1)
#     else:
#         EXAMPLE_FILE_NAME = sys.argv[4]
#
#         if os.path.isfile(EXAMPLE_FILE_NAME):
#             print("Complex change!")
#         else:
#             print("File %s does not exist in the path" %EXAMPLE_FILE_NAME)
#             print("Exitting...")
#             sys.exit(1)
#         # Check if the file exist first
#
# # for item in sys.argv[1:]:
# #     print(item)