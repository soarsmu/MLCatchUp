import sys

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
if len(sys.argv) != 2:
    print('usage: refer to help/readme. use DSL.py help for list of available commands')
    sys.exit(1)

with open(sys.argv[1], 'r') as file:
    for line in file:
        line = line.strip()
        if not line or line[0] == '#':
            continue
        parts = line.split()
        print(parts)