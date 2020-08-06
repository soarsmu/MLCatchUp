# Utility code to parse the input of the program
# Will need to parse the if then else input for the added constraint
import sys
import os

# Function to read the configuration file which define the transformation to be done
# TODO: IMPORTANT: THIS IS NOT NEEDED SINCE WE SHOULD BE ABLE TO DO AUTOMATIC INFERENCE
def read_config(filename):
    # open the file
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            print(f)

# Helper function to parse API signature, splitting it into three parts:
# 1. API/function name
# 2. Positional parameters
# 3. Keyword parameters
# For example:
# parse_api_signature("torch.div(param1:torch.IntTensor, param2:torch.IntTensor, *, out:torch.Tensor=None)")
# will return:
# API_Name = torch.div
# Positional_Parameter = [param1:torch.IntTensor, param2:torch.IntTensor]
# Keyword_Only_Parameter = [out:torch.Tensor=None]
# ':' signifies the type of the parameter
# :=: signifies the default value of the parameter
def parse_api_signature(api_string):
    # Find split index
    split_idx = api_string.find("(")
    api_name = api_string[0:split_idx]
    all_param = api_string[split_idx + 1 : -1]
    key_param = []
    split_param = all_param.split("*")
    pos_param = split_param[0].split(",")
    if len(split_param) > 1:
        key_param = split_param[1].split(",")
    stripped_pos_param = []
    stripped_key_param = []
    # Remove all whitespace from list
    for param in pos_param:
        if not(param.isspace() or not param):
            stripped_pos_param.append(param.strip())
    for param in key_param:
        if not(param.isspace() or not param):
            stripped_key_param.append(param.strip())

    print(api_name)
    print(stripped_pos_param)
    print(stripped_key_param)
    return api_name, stripped_pos_param, stripped_key_param

parse_api_signature("torch.div()")