# Utility code to parse the input of the program
# Will need to parse the if then else input for the added constraint
import sys
import os

# Class to parse the api parameter string and parse it accordingly
# e.g. api_parameter("param1:torch.IntTensor=torch.tensor(0)")
class api_parameter:
    param_type = ""
    param_default_value = ""
    param_name = ""

    def __str__(self):
        return self.param_name + ":" + self.param_type + "=" + self.param_default_value

    def __repr__(self):
        return self.__str__()

    def __init__(self, parameter_string):
        self.param_string = parameter_string
        self.parse_param_string(self.param_string)

    def parse_param_string(self, parameter_string):
        try:
            param_string = self.param_string
            # Get the type and value
            idx_value = param_string.find("=")
            idx_type  = param_string.find(":")
            if idx_type != -1 or idx_value != -1:
                # has type or value
                if idx_value != -1:
                    self.param_default_value = param_string[idx_value + 1:]
                    param_string = param_string[0:idx_value]
                if idx_type != -1:
                    self.param_type = param_string[idx_type + 1:]
                    param_string = param_string[0:idx_type]
            self.param_name = param_string.strip()
        except Exception as e:
            print("Problem in parsing the API parameter: " + self.param_string)
            print(e)

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

    # Convert all the API parameter into api_parameter class
    return_pos_param = []
    for param in stripped_pos_param:
        return_pos_param.append(api_parameter(param))

    return_key_param = []
    for param in stripped_key_param:
        return_key_param.append(api_parameter(param))

    return api_name, return_pos_param, return_key_param


# Helper function to list all the differences between old api and new api
# Return it in the form of list of DSL string that defines all the transformations
def list_all_differences(old_api, new_api):
    return []

# parse the API signature
# torch.div test
api_name, positional_param, keyword_param = parse_api_signature("torch.div(param1:torch.IntTensor, param2:torch.IntTensor, *, out:torch.Tensor=None)")
print(api_name)
print(positional_param)
print(keyword_param)
# torch.btrifact(A: Tensor, pivot=True, out=None)
# torch.lu(A, pivot=True, get_infos=False, out=None)