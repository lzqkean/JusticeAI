from src.fact_extraction.FactTree import LogicParser, WordStack
from src.fact_extraction.Models import logic, clause, predicate, compliment


class Proposition():
    #####################################
    # CONSTRUCTOR
    def __init__(self):
        self.__proposition_lst = []
        self.__stack = WordStack.Stack()
        self.__predicates = LogicParser.Tree()

    #####################################
    # RESET
    def __reset(self):
        self.__proposition_lst = []
        self.__stack.clear()

    #############################################
    # BUILD
    # -------------------------------------------
    # Parses sentence into logical objects
    # clause / predicates
    # From the list of logical objects, make
    # correlation between them
    #
    # sentence: string
    # draw: boolean
    def build(self, sentence, draw = False):
        self.__reset()
        self.__predicates.build(sentence, draw)
        predicate_lst = self.__predicates.get_logic_model()
        self.__create_logic(predicate_lst)

    #############################################
    # CREATE LOGIC
    # -------------------------------------------
    # Will perform correct operation depending on
    # logical object type
    # clause, predicate, or compliment
    #
    # logic_lst: list
    def __create_logic(self, logic_lst):
        for i in range(len(logic_lst)):
            if type(logic_lst[i]) == clause.Clause:
                self.__clause_operation(logic_lst[i], logic_lst, i)
            elif type(logic_lst[i]) == predicate.Predicate:
                self.__predicate_operation(logic_lst[i], logic_lst, i)
            elif type(logic_lst[i]) == compliment.compliment:
                self.__compliment_operation(logic_lst[i], logic_lst, i)

    #############################################
    # CLAUSE OPERATION
    # -------------------------------------------
    # 1- If no predicate then append to clause list
    # 2- If predicate before and after the word then
    #    create relationship and append word to clause stack
    # 3- else append to compliment stack
    # 4- if last word in the list then extract features
    #
    # logic: Model.AbstractModel
    # logic_lst: list[Model.AbstractMode]
    # index: integer
    def __clause_operation(self, logic, logic_lst, index):
        if self.__stack.peek_predicate() is None:
            self.__stack.clause_stack.append(logic)

        elif type(self.__stack.next(logic_lst, index)) == predicate.Predicate:
            if type(self.__stack.previous(logic_lst, index)) == predicate.Predicate:
                self.__stack.compliment_stack.append(logic)
            self.__extract_relations()
            self.__stack.clause_stack.append(logic)
            return

        else:
            self.__stack.compliment_stack.append(logic)

        if self.__stack.next(logic_lst, index) is None:
            self.__extract_relations()

    #############################################
    # PREDICATE OPERATION
    # -------------------------------------------
    # 1- if no predicate in stack then append predicate
    # 2- else pop predicate and merge them into 1 phrase
    #    append new predicate
    # 3- if last word in list then extract features
    #
    # logic: Model.AbstractModel
    # logic_lst: list[Model.AbstractMode]
    # index: integer
    def __predicate_operation(self, logic_model, logic_lst, index):
        if self.__stack.peek_predicate() is None:
            self.__stack.predicate_stack.append(logic_model)

        else:
            model = self.__stack.predicate_stack.pop()
            model.merge(logic_model)
            self.__stack.predicate_stack.append(model)

        if self.__stack.next(logic_lst, index) is None:
            self.__extract_relations()

    #############################################
    # COMPLIMENT OPERATION
    # -------------------------------------------
    # 1- if last word in list then extract features
    # 2- if word in between 2 predicates then extract
    #    features and append word to clause
    # 3- else append to compliment stack
    #
    # logic: Model.AbstractModel
    # logic_lst: list[Model.AbstractMode]
    # index: integer
    def __compliment_operation(self, logic, logic_lst, index):
        if self.__stack.next(logic_lst, index) is None:
            self.__stack.compliment_stack.append(logic)
            self.__extract_relations()

        elif type(self.__stack.next(logic_lst, index)) == predicate.Predicate:
            if type(self.__stack.previous(logic_lst, index)) == predicate.Predicate:
                self.__stack.compliment_stack.append(logic)
            self.__extract_relations()
            self.__stack.clause_stack.append(logic)

        else:
            self.__stack.compliment_stack.append(logic)

    #############################################
    # EXTRACT RELATIONS
    # -------------------------------------------
    # 1- Pop predicate
    # 2- For ever clause map them to their compliments
    # 3- clear stack
    def __extract_relations(self):
        predicate = self.__stack.predicate_stack.pop()
        for clause in self.__stack.clause_stack:
            for compliment in self.__stack.compliment_stack:
                model = logic.LogicModel()
                model.clause = clause
                model.predicate = predicate
                model.compliment = compliment
                self.__proposition_lst.append(model)
        self.__stack.clear()

    def get_proposition_lst(self):
        return self.__proposition_lst.copy()


if __name__ == "__main__":
    p = Proposition()
    p.build("Though he was very rich, he was still very unhappy.")
    lst = p.get_proposition_lst()
    for e in lst:
        print(e)
        print()