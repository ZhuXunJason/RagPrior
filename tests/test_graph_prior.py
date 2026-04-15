import pytest
from coverage_ranker import CoverageGroup, rank_groups
from kg_ontology import build_ontology_features, OntologyFeatures
from report_utils import validate_full_permutation

def test_ontology_feature_attach():
    features = build_ontology_features(
        operation="login",
        page="home",
        element="button",
        bug="crash",
        description="I am stuck on home"
    )
    assert "auth_action" in features.operation_families
    assert "main" in features.modules
    assert "main" in features.page_types
    assert "interactive" in features.element_types
    assert "fatal" in features.symptom_families

def test_coverage_greedy_ranking():
    g1 = CoverageGroup(
        group_id=1,
        representative_report_num=1,
        member_report_nums=[],
        ontology_features=OntologyFeatures(
            operation_families={"auth_action"},
            symptom_families={"fatal"},
            page_types=set(),
            modules=set(),
            element_types=set()
        )
    )
    g2 = CoverageGroup(
        group_id=2,
        representative_report_num=2,
        member_report_nums=[],
        ontology_features=OntologyFeatures(
            operation_families={"click_action"},
            symptom_families={"fatal"},
            page_types=set(),
            modules=set(),
            element_types=set()
        )
    )
    g3 = CoverageGroup(
        group_id=3,
        representative_report_num=3,
        member_report_nums=[],
        ontology_features=OntologyFeatures(
            operation_families={"auth_action"},
            symptom_families={"fatal"},
            page_types=set(),
            modules=set(),
            element_types=set()
        )
    )
    ranked = rank_groups([g1, g2, g3])
    assert ranked[0].representative_report_num == 2
    assert ranked[1].representative_report_num == 1
    assert ranked[2].representative_report_num == 3

def test_final_sequence_legality():
    from report_utils import validate_full_permutation
    assert validate_full_permutation([1, 2, 3], 3)
    assert not validate_full_permutation([1, 2], 3)
    assert not validate_full_permutation([1, 2, 2], 3)
    assert not validate_full_permutation([1, 2, 4], 3)

def test_no_severity_dependency():
    pass
