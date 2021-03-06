import case
import os

case = case.Case(os.getcwd() + "/feature_extraction/dummy_cases/sample_case1.txt")


def test_no_demande():
    assert len(case.data["applicationID"]) == 1
    assert case.data["applicationID"][0] == "1234567"


def test_no_dossier():
    assert len(case.data["fileID"]) == 1
    assert case.data["fileID"][0] == "123456 12 12345678 D"


def test_total_hearings():
    assert case.data["totalHearings"] == 2


def test_is_rectified():
    assert case.data["isRectified"]
