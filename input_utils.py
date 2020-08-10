# Utility code to parse the input of the program
# Will need to parse the if then else input for the added constraint
import sys
import os

# Class to parse the api parameter string and parse it accordingly
# e.g. api_parameter("param1:torch.IntTensor=torch.tensor(0)")
class ApiParameter:
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

# Class to keep API invocation that has separated API name, positional parameter, and keyword parameter
class ApiSignature:
    api_name = ""
    positional_param = []
    keyword_param = []
    def __init__(self, api_name, pos_param, key_param):
        self.api_name = api_name
        self.positional_param = pos_param
        self.keyword_param = key_param

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Api Name: " + self.api_name + "\nPositional Param: " + self.positional_param.__str__() + "\nKeyword Param: " + self.keyword_param.__str__() + "\n"


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

    # print(api_name)
    # print(stripped_pos_param)
    # print(stripped_key_param)

    # Convert all the API parameter into api_parameter class
    return_pos_param = []
    for param in stripped_pos_param:
        return_pos_param.append(ApiParameter(param))

    return_key_param = []
    for param in stripped_key_param:
        return_key_param.append(ApiParameter(param))

    return_signature = ApiSignature(api_name, return_pos_param, return_key_param)
    return return_signature


# Check if two API parameter is the same
# Return True if they are the same
# Return False if they are not
def is_same_param(param1:ApiParameter, param2:ApiParameter):
    if param1.param_name == param2.param_name:
        # if name is same, there is a chance that they are the same
        if not param1.param_type or not param2.param_type:
            # if either is none and have the same name, return true
            return True
        elif param1.param_type == param2.param_type:
            # if the type is the same, check the default value
            if not param1.param_default_value or not param2.param_default_value:
                return True
            elif param1.param_default_value == param2.param_default_value:
                return True
            else:
                return False
        else:
            # different type
            return False


    else:
        return False

# Remove same parameter from the old api and new api
# Help in removing the unnecessary parameter
def eliminate_same_param(old_api: ApiSignature, new_api: ApiSignature):
    old_pos_param = old_api.positional_param
    old_key_param = old_api.keyword_param
    new_pos_param = new_api.positional_param
    new_key_param = new_api.keyword_param

    # Eliminate the positional param first
    i = 0
    while i < len(old_pos_param):
        i_param = old_pos_param[i]
        same_param = False
        for j in range(0, len(new_pos_param)):
            j_param = new_pos_param[j]
            if is_same_param(i_param, j_param):
                # Same param detected
                # Remove both
                old_pos_param.pop(i)
                new_pos_param.pop(j)
                same_param = True
                break
        if not same_param:
            i += 1

    # Then eliminate the keyword param too
    i = 0
    while i < len(old_key_param):
        i_param = old_key_param[i]
        same_param = False
        for j in range(0, len(new_key_param)):
            j_param = new_key_param[j]
            if is_same_param(i_param, j_param):
                # Same param detected
                # Remove both
                old_key_param.pop(i)
                new_key_param.pop(j)
                same_param = True
                break
        if not same_param:
            i += 1
    old_api.positional_param = old_pos_param
    old_api.keyword_param = old_key_param
    new_api.positional_param = new_pos_param
    new_api.keyword_param = new_key_param


# Helper function to list all the differences between old api and new api
# Return it in the form of list of DSL string that defines all the transformations
def list_all_differences(old_api: ApiSignature, new_api: ApiSignature):
    list_differences = []
    # First, check name
    old_name, new_name = old_api.api_name, new_api.api_name

    if old_name != new_name:
        # Add update name query
        list_differences.append("RENAME_API " + old_name + " TO " + new_name)

    # Then check the parameter

    return []

# parse the API signature
# torch.div test
torch_div_signature = parse_api_signature("torch.div(param1:torch.IntTensor, param2:torch.IntTensor, *, out:torch.Tensor=None)")

# torch.btrifact(A: Tensor, pivot=True, out=None)
old_signature = parse_api_signature("torch.btrifact(A: Tensor, pivot=True, out=None)")

# torch.lu(A, pivot=True, get_infos=False, out=None)
new_signature = parse_api_signature("torch.lu(A, pivot=True, get_infos=False, out=None)")

print("Signature")
print(old_signature)
print(new_signature)

eliminate_same_param(old_signature, new_signature)
