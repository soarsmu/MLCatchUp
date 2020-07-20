import ast
from ast import *
from pprint import pprint
import astunparse

# Documentation for Python AST in the following links:
# 1. https://greentreesnakes.readthedocs.io/en/latest/index.html
# 2. https://kite.com/python/docs/ast.NodeVisitor

# Things to remember when creating a new node:
# 1. A node must have lineno and col_offset attributes

# Some of the hurdles in AST modification in Python that do not exist in Java:
# 1. Dynamic typing
# 2. No usage of semicolon that can be used to determine a statement ending
# 3. No usage of brackets that are instead changed into indentation. These indentation makes it harder to create
#    a new node because we will need to determine the line number and the offset of such new node.
from Python_AST_Test.ast_utils import *
from Python_AST_Test.ast_transform_rule import *
from Python_AST_Test.api_formatter import *
from Python_AST_Test.pipeline_utils import *

# This can and should be changed into input parameter for the program
DEPRECATED_API_NAME = "sklearn.cluster.KMeans"
DEPRECATION_TYPE = "remove_keyword_param"
DEPRECATED_API_KEYWORD = ["n_jobs"]
default_test = "ast_test.py"
sklearn_dummyclassifier_file = "../sklearn_testcases/sklearn_dummyclassifier.py"

# clone git repo
repo_list = [
"git://github.com/PranaliKanade-28/vidhi.git",
"git://github.com/bentsherman/ml-toolkit.git",
"git://github.com/39239580/dev_ops_recom_sys_ser.git",
"git://github.com/bharlow058/Imbalance-Library.git",
"git://github.com/raafatzahran/Udacity-DataScience.git",
"git://github.com/cmdydd/server.git",
"git://github.com/fragiacalone/project_image.git",
"git://github.com/angelfish91/domain_regex_extract.git",
"git://github.com/tychen5/ML_DL_project.git",
"git://github.com/Viniciuspsc/Unsupervised-learning-Python.git",
"git://github.com/capissimo/Machine-Learning-with-Python-Cookbook.git",
"git://github.com/zegamezhees001/Project.git",
"git://github.com/tasrif60/final_project.git",
"git://github.com/VinceJiale/ml-with-python-cookbook.git",
"git://github.com/PolarisZhang/FE595_Assignment-4.git",
"git://github.com/binfalc/Machine-Learning.git",
"git://github.com/kongxiangshi/machine_learning.git",
"git://github.com/masa-kato/Metis-Work.git",
"git://github.com/NLPatVCU/Sentiment-Classification.git",
"git://github.com/fcavani/pgm.git",
"git://github.com/centre-for-humanities-computing/corona-psychopathology.git",
"git://github.com/timothyjlaurent/ga_tech_dimensional_reduction.git"
]


for repo in repo_list:
    # folder_name = clone_repo("git://github.com/Gavin16/Albb-TC-Competition.git")
    folder_name = clone_repo(repo)
    print(folder_name)
    if folder_name is None:
        continue

    # Process all file and printing the result
    list_file = list_all_file(folder_name)
    with open(folder_name + "/result.txt", "w", encoding="utf-8") as writefile:
        writefile.write("Repo source: " + repo.__str__() + "\n")
        for file in list_file:
            list_of_list_changes = process_file(file, DEPRECATED_API_NAME,
                                                DEPRECATION_TYPE, keywords=DEPRECATED_API_KEYWORD)
            writefile.write("Processing file: " + file + "\n")
            if list_of_list_changes is not None:
                for list_changes in list_of_list_changes:
                    for str in list_changes:
                        writefile.write(str + "\n")
            writefile.write("----------------------------------\n")

    writefile.close()


# with open(sklearn_dummyclassifier_file, "r", encoding="utf-8") as source:
# # with open(default_test, "r") as source:
#     tree = ast.parse(source.read())

# Don't need the process api_format I think
# Maybe let's make a new function called get all function alias that get all the alias of the deprecated API?
# Or maybe, we can just forget all about the transformer and simply walk the ast?
# I think it is better to get all the function alias?
# Maybe just process the import and import from but not process the assign
# list_api, import_dict, from_import_dict = process_api_format(tree)
# list_deprecated_api = []
# list_line_number = []
# This is to filter the name
# Need further filter depending on the Deprecation type. May need to check the keyword param
# for api in list_api:
#     current_keys = api["key"]
#     if DEPRECATED_API_NAME in api["name"]:
#         key_is_correct = True
#         for key in DEPRECATED_API_KEYWORD:
#             if key not in current_keys:
#                 key_is_correct = False
#         if key_is_correct:
#             list_deprecated_api.append(api)
#
# for api in list_deprecated_api:
#     print(api)
#     list_line_number.append(api["line_no"])
#
#
# print("Import dict: " + import_dict.__str__())
# print("From Import dict: " + from_import_dict.__str__())
# paramRemove = KeywordParamRemover("KMeans", "n_jobs", list_line_number)
# paramRemove.transform(tree)




# defaultTransformer = DefaultParamValueTransformer("KMeans", "n_clusters", 1)
# defaultTransformer.transform(tree)

# nameTransformer = ApiNameTransformer("KMeans", "KNotMeans")
# nameTransformer.transform(tree)

# defaultValueChange = DefaultParamValueTransformer("DummyClassifier", "strategy", "stratified")
# defaultValueChange.transform(tree)
#
# apiNameChange = ApiNameTransformer("DummyClassifier", "StupidClassifier")
# apiNameChange.transform(tree)
