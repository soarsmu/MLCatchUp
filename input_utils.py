# Utility code to parse the input of the program
# Will need to parse the if then else input for the added constraint
import sys
import os
from DSL import *
import re

# Class to parse the api parameter string and parse it accordingly
# e.g. api_parameter("param1:torch.IntTensor=torch.tensor(0)")
class ApiParameter:
    param_type = ""
    param_default_value = ""
    param_name = ""
    position = -1

    def __str__(self):
        return self.param_name + ":" + self.param_type + "=" + self.param_default_value

    def __repr__(self):
        return self.__str__()

    def __init__(self, parameter_string, position=-1):
        self.param_string = parameter_string
        self.parse_param_string(self.param_string)
        self.position = position

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
    # Add the position
    i = 1
    for param in stripped_pos_param:
        return_pos_param.append(ApiParameter(param, i))
        i += 1

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


# Get the api mapping between the leftover old api parameter and the new api parameter
# The targeted functionality is that the mapping will be inferred automatically
# return a dictionary mapping between the old api parameter with the new api parameter
# USABLE FOR CHANGE PARAMETER NAME DEPRECATION
def get_api_mapping(old_api: ApiSignature, new_api: ApiSignature):
    return_dict = {}
    old_pos_param = old_api.positional_param
    old_key_param = old_api.keyword_param
    new_pos_param = new_api.positional_param
    new_key_param = new_api.keyword_param

    # TODO: make an advanced API mapper which may incorporate NLP / semantic understanding
    # Map based on type first

    # Map the positional param first
    i = 0
    while i < len(old_pos_param):
        i_param = old_pos_param[i]
        same_param = False
        for j in range(0, len(new_pos_param)):
            j_param = new_pos_param[j]
            if i_param.param_type == j_param.param_type:
                # Same param detected
                # Remove both
                old_pos_param.pop(i)
                new_pos_param.pop(j)
                same_param = True
                return_dict[i_param] = j_param
                break
        if not same_param:
            i += 1
    print("Api Mapping Dictionary")
    print(return_dict)
    return return_dict


# Get the API mapping between the leftover old api positional parameter with the new api keyword parameter
# It is used to find the type of deprecation in which positional parameter is transformed into keyword parameter
# USABLE FOR POSITIONAL TO KEYWORD API DEPRECATION
def get_positional_to_keyword_param(old_api: ApiSignature, new_api: ApiSignature):
    return_dict = {}
    old_pos_param = old_api.positional_param
    new_key_param = new_api.keyword_param
    i = 0
    while i < len(old_pos_param):
        i_param = old_pos_param[i]
        same_param = False
        for j in range(0, len(new_key_param)):
            j_param = new_key_param[j]
            if i_param.param_type == j_param.param_type or i_param.param_name == j_param.param_name:
                # Same param detected
                # Remove both
                old_pos_param.pop(i)
                new_key_param.pop(j)
                same_param = True
                return_dict[i_param] = j_param
                break
        if not same_param:
            i += 1
    print("Positional to Keyword dictionary")
    print(return_dict)
    return return_dict

# Helper function to list all the differences between old api and new api
# Return it in the form of list of DSL string that defines all the transformations
def list_all_differences(old_api: ApiSignature, new_api: ApiSignature):
    list_differences = []
    # First, check name
    old_name, new_name = old_api.api_name, new_api.api_name

    if new_name == "":
        dsl = "REMOVE_API " + old_name
        list_differences.append(dsl)
        return list_differences

    # Then check the parameter
    # Remove same parameter first
    print("Signature")
    print(old_api)
    print(new_api)

    eliminate_same_param(old_api, new_api)

    print("Signature After elimination")
    print(old_api)
    print(new_api)

    # 1. Get positional param to keyword param transformation
    positional_to_keyword_dict = get_positional_to_keyword_param(old_api, new_api)
    for key, value in positional_to_keyword_dict.items():
        if key.position > 0:
            dsl = "POSITIONAL_TO_KEYWORD POSITION " + key.position.__str__() + " KEYWORD " + value.param_name + " FOR " + old_name
            list_differences.append(dsl)

    # 2. Approximate name change for the parameter
    get_api_mapping(old_api, new_api)
    # Process the API name change mapping
    

    # 3. Process leftover positional parameter and keyword parameter from old API
    #    The leftovers should be deleted
    list_deleted_pos_param = old_api.positional_param
    print("Deleted pos param")
    print(list_deleted_pos_param)
    # 4. Also process the leftover keyword parameter
    list_deleted_key_param = old_api.keyword_param
    print("Deleted key param")
    print(list_deleted_key_param)
    for param in list_deleted_key_param:
        removed_name = param.param_name
        dsl = "REMOVE_KEYWORD_PARAM " + removed_name + " FOR " + old_name
        list_differences.append(dsl)

    # 5. Then, process the leftovers parameter from the new API
    #    The leftovers from the new API should be added (e.g. a new default parameter)
    list_new_param = new_api.positional_param + new_api.keyword_param
    print("Added param")
    print(list_new_param)
    for param in list_new_param:
        # If has default value
        if param.param_default_value:
            dsl = "ADD_PARAM " + param.param_name + " WITH_VALUE " + param.param_default_value + " FOR " + old_name
            list_differences.append(dsl)

    print("OLDIE")
    print(old_name)
    print(new_name)

    if old_name != new_name:
        # Add update name query
        list_differences.append("RENAME_API " + old_name + " TO " + new_name)
    print("DSL LIST: ")
    print(list_differences)


    return list_differences


# Input: Transformation dictionary from the get_list_diff function and the filename
# Output: file is transformed
def apply_transformation(transformation_dictionary, filename):
    list_position = list(transformation_dictionary.keys())
    print(list_position)
    # Read the file into lines
    file_line_list = []
    with open(filename, "r", encoding="utf-8") as f:
        file_line_list = f.readlines()
        f.close()
    # Traverse in reverse to conduct the transformation

    for i, line in reversed(list(enumerate(file_line_list))):
        # Special case if in last line
        if i == (len(file_line_list) - 1) and (i + 1) in list_position:
            print("Last line")
            new_value = ""
            old_value, new_value_list = transformation_dictionary[i + 1]
            for value in new_value_list:
                new_value = new_value + value + "\n"
            file_line_list[i] = new_value
        # Special case if index is one
        elif i == 1 and i in list_position:
            print("Special case for from import")
            old_value, new_value_list = transformation_dictionary[i]
            new_value = ""
            for value in new_value_list:
                new_value = new_value + value + "\n"
            file_line_list.insert(0, new_value)
        # if current index is available in list position
        elif i in list_position:
            print("Replacing the value")
            print(file_line_list[i])
            old_value, new_value_list = transformation_dictionary[i]
            new_value = ""

            # Check multiline to make sure that the total of the multiline makes the old value. Use strip!
            old_value = re.sub('[()]', '', "".join(old_value.split()))
            current_value = re.sub('[()]', '', "".join(file_line_list[i - 1].split()))
            num_to_delete = 0
            # Change the method into removing the old value little by little
            while len(old_value) > 0:
                print("Masuk while")
                print(old_value)
                print(current_value)
                # Should remove comments from current_value
                index_comment = current_value.find('#')
                if index_comment != -1:
                    current_value = current_value[:index_comment]
                if current_value in old_value:
                    old_value = old_value.replace(current_value, '')
                    try:
                        current_value = re.sub('[()]', '', "".join(file_line_list[i + num_to_delete].split()))
                    except:
                        # If except, probably the last line of the file
                        current_value = ""
                    num_to_delete += 1
                else:
                    # Special case if the leftover is zero
                    if old_value == "0":
                        break
                    if num_to_delete > 0:
                        num_to_delete += 1
                    break

            print("Num to delete:")
            print(num_to_delete)
            # while current_value != old_value:
            #     print("HERE")
            #     print(current_value)
            #     print(old_value)
            #     current_value = current_value + re.sub('[()]', '', "".join(file_line_list[i + num_to_delete].split()))
            #     num_to_delete += 1
            while num_to_delete > 1:

                popped = file_line_list.pop(i)
                print("Popped")
                print(popped)
                num_to_delete -= 1
            for value in new_value_list:
                new_value = new_value + value + "\n"

            # Fixing indentation here
            # Get previous line
            previous_line = file_line_list[i - 1]
            # Get the actual indentation
            actual_indentation = re.match(r"\s*", previous_line).group()
            if new_value != "":
                new_value = actual_indentation + new_value.lstrip()

            file_line_list[i - 1] = new_value
            print("Updating value:")
            print(file_line_list[i - 1])

    with open("updated_" + filename, "w", encoding="utf-8") as f:
        print("Before printing the updated file")
        for line in file_line_list:
            f.write(line)
        f.close()