import pytest
from src.inference import predict_match, get_matcher_instance


def test_matcher_instance_creation():
    matcher = get_matcher_instance()
    assert matcher is not None
    assert matcher.model_type in ["distilbert", "logistic_regression", "distilbert_pretrained"]


def test_predict_match_structure():
    resume = "Senior Python Developer with 5 years experience in Django, REST APIs, and Docker."
    jd = "Looking for Senior Backend Developer with Python, Django, Docker, and PostgreSQL."

    result = predict_match(resume, jd)

    assert "label" in result
    assert result["label"] in ["Good Fit", "Potential Fit", "No Fit"]
    assert "score" in result
    assert 0.0 <= result["score"] <= 1.0
    assert "score_percentage" in result
    assert "%" in result["score_percentage"]
    assert "probabilities" in result
    assert len(result["probabilities"]) == 3

    # Check that probabilities sum to approximately 1.0
    prob_sum = sum(result["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.05

    # Check skill analysis structure
    assert "skill_analysis" in result
    assert "matching_skills" in result["skill_analysis"]
    assert "missing_skills" in result["skill_analysis"]
    assert "recommendations" in result["skill_analysis"]


def test_predict_match_empty_input_raises_error():
    with pytest.raises(ValueError):
        predict_match("", "Valid JD")

    with pytest.raises(ValueError):
        predict_match("Valid Resume", "")
