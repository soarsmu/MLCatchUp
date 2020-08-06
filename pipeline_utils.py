import os
import ast
from api_formatter import *
from ast_transform_rule import *
from git import Repo


# Input  : a string of git repository
# Output : the name of the newly created/cloned repository for processing
def clone_repo(git_repo_url):
    repo_name = "../sklearn_testcases/" + git_repo_url.split("/")[-1]
    repo_name = repo_name[0:repo_name.rfind('.')]
    print("Cloning Repository: " + repo_name)
    try:
        Repo.clone_from(git_repo_url, repo_name)
    except Exception as e:
        print(e)
        print("Error, repo already exist")
        return repo_name
    return repo_name

# Input  : a folder path string to be processed recursively
# Output : return a list of python file within the folder and its subdirectories
# Output : may also output the list of file
def list_all_file(folder_path):
    # list of the name that should not be processed
    list_not_processed = [".git", "venv", "site-packages"]
    list_of_file = []
    walk_dir = folder_path

    for root, subdirs, files in os.walk(walk_dir):
        # Does not need to list subdirectory
        # for subdir in subdirs:
        #     print('\t- subdirectory ' + subdir)

        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = file_path[file_path.rfind(".") + 1:]
            if file_extension == "py":
                contain_not_processed = False
                for term in list_not_processed:
                    if term in file_path:
                        contain_not_processed = True
                        break
                if not contain_not_processed:
                    list_of_file.append(file_path)
    return list_of_file

# Input:
# 1. File path - Path towards the file to be processed
# 2. deprecated API name
# 3. deprecation type (e.g. parameter removal)
# 4. kwargs based on the deprecation type
# Output:
# A list of string that defines the change to the updated file
def process_file(file_path, api_name, deprecation_type, **kwargs):
    # General processing first
    # Open the file and list the deprecation
    source = open(file_path, "r", encoding="utf-8")
    # with open(default_test, "r") as source:
    listChange = []
    list_deprecated_api = []
    list_line_number = []
    list_keywords = kwargs.get("keywords", [])
    try:
        tree = ast.parse(source.read())
    except Exception as e:
        print(e)
        return
    list_api, import_dict, from_import_dict = process_api_format(tree)
    # This is to filter the name of the api to make sure only the deprecated API is processed
    # Need further filter depending on the Deprecation type. May need to check the keyword param
    for api in list_api:
        current_keys = api["key"]
        if api_name in api["name"]:
            key_is_correct = True
            for key in list_keywords:
                if key not in current_keys:
                    key_is_correct = False
            if key_is_correct:
                list_deprecated_api.append(api)

    if len(list_deprecated_api) > 0:
        listChange.append(["List of deprecated APIs detected: "])
        for api in list_deprecated_api:
            list_line_number.append(api["line_no"])
            listChange.append(["Line Number: " + api["line_no"].__str__()])
            listChange.append(["API Name: " + api["name"]])
        listChange.append(["\n"])
        if deprecation_type == "remove_keyword_param":
            for key in list_keywords:
                paramRemove = KeywordParamRemover(api_name, key, list_line_number)
                paramRemove.transform(tree)
                listChange.append(paramRemove.listChanges)
        source.close()
    return listChange


# TODO: ast module does not provide automated application of the change
# Furthermore, it also removes all comments and formatting in the code.
# Therefore, to apply, its better to just list the change first and then
# do basic line by line file formatting.
def apply_changes():
    print("TODO")






# Input  : a folder path string to be processed recursively
# Output : return a list of python file within the folder and its subdirectories
# Output : may also output the list of file
def list_all_file_nofilter(folder_path):
    # list of the name that should not be processed
    list_not_processed = []
    list_of_file = []
    walk_dir = folder_path

    for root, subdirs, files in os.walk(walk_dir):
        # Does not need to list subdirectory
        # for subdir in subdirs:
        #     print('\t- subdirectory ' + subdir)

        for filename in files:
            file_path = os.path.join(root, filename)
            file_extension = file_path[file_path.rfind(".") + 1:]
            if file_extension == "py":
                contain_not_processed = False
                for term in list_not_processed:
                    if term in file_path:
                        contain_not_processed = True
                        break
                if not contain_not_processed:
                    list_of_file.append(file_path)
    return list_of_file