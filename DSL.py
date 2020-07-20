import sys
import os

# Tutorial based on this article:
# https://dbader.org/blog/writing-a-dsl-with-python

# 1. Change parameter name
# 2. Change positional parameter into keyword parameter
# 3. Remove parameter
# 4. Change method name
# 5. Complex change


# 1:
"change_param_name fully_qualified_api_name old_param_name new_param_name filename"

# 2:
"positional_to_keyword fully_qualified_api_name parameter_position keyword_param_name filename"

# 3:
"remove_param fully_qualified_api_name keyword_param_name filename"
"remove_param fully_qualified_api_name parameter_position filename"

# 4:
"change_method_name old_fully_qualified_name new_fully_qualified_name filename"

# 5:
"complex_transform filename example_filename"

# dsl1.py


# The source file is the 1st argument to the script
if len(sys.argv) < 4:
    print('usage: refer to help/readme. use DSL.py help for list of available commands')
    sys.exit(1)

DSL_MODE = sys.argv[1]
ORIGINAL_API_NAME = sys.argv[2]
FILENAME = sys.argv[3]

if DSL_MODE == "change_param_name":
    # Change param name
    # Must have 4 additional arguments (sys.argv len must be 6)
    if len(sys.argv) != 6:
        print("usage: change_param_name fully_qualified_api_name filename old_param_name new_param_name")
        sys.exit(1)
    else:
        print("Changing parameter name")
        OLD_PARAM_NAME = sys.argv[4]
        NEW_PARAM_NAME = sys.argv[5]


if DSL_MODE == "positional_to_keyword":
    # Positional parameter to keyword parameter
    # Must have 4 additional arguments (sys.argv len must be 6)
    if len(sys.argv) != 6:
        print("usage: positional_to_keyword fully_qualified_api_name filename parameter_position keyword_param_name")
        sys.exit(1)
    else:
        try:
            PARAMETER_POSITION = int(sys.argv[4])
        except Exception:
            print("usage: positional_to_keyword fully_qualified_api_name filename parameter_position keyword_param_name")
            sys.exit(1)
        KEYWORD_PARAMETER_NAME = sys.argv[5]
        print("Converting positional keyword")


if DSL_MODE == "remove_param":
    # Remove parameter
    # Must have 3 additional arguments (sys.argv len must be 5)
    if len(sys.argv) != 5:
        print("usage: remove_param fully_qualified_api_name filename keyword_param_name")
        print("usage: remove_param fully_qualified_api_name filename positional_param_position")
        sys.exit(1)
    else:
        if sys.argv[4].isnumeric():
            # removal of positional parameter
            REMOVED_PARAM = int(sys.argv[4])
        else:
            REMOVED_PARAM = sys.argv[4]
        print("Removing parameter")

if DSL_MODE == "change_method_name":
    # Change method name
    # Must have 3 additional arguments (sys.argv len must be 5)
    if len(sys.argv) != 5:
        print("usage: change_method_name filename old_fully_qualified_name new_fully_qualified_name")
        sys.exit(1)
    else:
        NEW_API_NAME = sys.argv[4]
        print("Changing method name")

if DSL_MODE == "complex_transform":
    # Complex transformation
    # Must have 3 additional arguments (sys.argv len must be 5)
    if len(sys.argv) != 5:
        print("usage: complex_transform fully_qualified_name filename example_filename")
        sys.exit(1)
    else:
        EXAMPLE_FILE_NAME = sys.argv[4]

        if os.path.isfile(EXAMPLE_FILE_NAME):
            print("Complex change!")
        else:
            print("File %s does not exist in the path" %EXAMPLE_FILE_NAME)
            print("Exitting...")
            sys.exit(1)
        # Check if the file exist first

# for item in sys.argv[1:]:
#     print(item)