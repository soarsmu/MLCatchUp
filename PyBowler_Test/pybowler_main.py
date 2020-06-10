# try:
#     import sh
# except ImportError:
#     # fallback: emulate the sh API with pbs
#     import pbs
#     class Sh(object):
#         def __getattr__(self, attr):
#             return pbs.Command(attr)
#     sh = Sh()
from bowler import Query
from fissix import pytree


path = "../testcases"

# Term and arithmetic operator seems to be different. In this case, term is used for division
# Query to convert term, default to / (addition) operator
def term_query(*, operator="/"):
    def change_operator(node, capture, filename):
        print(node)
        arr_children = node.children
        for child in arr_children:
            print(child)
            if child == '/':
                print("CHILD IS A SLASHER")
        # iterator = node.pre_order()
        # print(next(iterator))
        # print(next(iterator))
        # print(next(iterator))
        # print(next(iterator))
        # print(node.pre_order())
        # print(capture)
        # print(capture['node'])

    PATTERN = """
        term< any * >
    """
    (
        # Query("testcases").select(PATTERN).dump()
        Query("testcases").select(PATTERN).modify(change_operator).diff()
    )


def div_query():
    print("div_query")

def main():
    print("I WANNA CRY")
    term_query()
main()