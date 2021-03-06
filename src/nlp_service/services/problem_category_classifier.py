# -*- coding: utf-8 -*-
from services.gram_classifier import GramClassifier
import os


class ProblemCategoryClassifier(GramClassifier):
    inputFiles = ['deposits', 'lease_termination', 'nonpayment', 'rent_change']

    """docstring for ProblemCategoryClassifier"""

    def __init__(self, forceTrain=False):
        baseName = os.path.basename(__file__).split(".")[0]
        super().__init__(baseName,
                         ProblemCategoryClassifier.inputFiles,
                         forceTrain)

    def classify(self, questionInput):
        output = super().classify(questionInput)
        return {'category': output}
