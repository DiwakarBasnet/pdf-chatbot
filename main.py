from model import LLMSingleton
from db_class import DataBase
from util_func import query_rag, database_manipulation

def main():
    flag = True
    db = DataBase()
    database_manipulation(db, flag)
    input_query = input("Enter your query text: ")
    model = LLMSingleton()
    print(query_rag(model, input_query))

if __name__ == "__main__":
    main()

