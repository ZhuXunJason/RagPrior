from dataclasses import dataclass
from typing import List, Set, Tuple, Dict
import math

from kg_ontology import OntologyFeatures


@dataclass
class CoverageGroup:
    group_id: int
    representative_report_num: int
    member_report_nums: List[int]
    ontology_features: OntologyFeatures


def flatten_features(group: CoverageGroup) -> Set[Tuple[str, str]]:
    """Flatten OntologyFeatures into a set of (feature_type, feature_value) tuples."""
    flat_features = set()
    if not group.ontology_features:
        return flat_features
    
    for val in group.ontology_features.operation_families:
        flat_features.add(("operation_family", val))
    for val in group.ontology_features.symptom_families:
        flat_features.add(("symptom_family", val))
    for val in group.ontology_features.page_types:
        flat_features.add(("page_type", val))
    for val in group.ontology_features.modules:
        flat_features.add(("module", val))
    for val in group.ontology_features.element_types:
        flat_features.add(("element_type", val))
        
    return flat_features


def compute_group_idf(groups: List[CoverageGroup]) -> Dict[Tuple[str, str], float]:
    """Compute IDF weights for each unique feature across groups."""
    total_groups = len(groups)
    if total_groups == 0:
        return {}
        
    df_map: Dict[Tuple[str, str], int] = {}
    for group in groups:
        features = flatten_features(group)
        for feat in features:
            df_map[feat] = df_map.get(feat, 0) + 1
            
    idf_map: Dict[Tuple[str, str], float] = {}
    for feat, df in df_map.items():
        idf_map[feat] = math.log(total_groups / max(df, 1))
        
    return idf_map


def marginal_gain(group: CoverageGroup, covered: Set[Tuple[str, str]], idf_map: Dict[Tuple[str, str], float]) -> float:
    """Calculate the marginal gain of adding this group based on new features covered."""
    gain = 0.0
    group_features = flatten_features(group)
    
    if not group_features:
        return gain
        
    for feat in group_features:
        if feat not in covered:
            gain += idf_map.get(feat, 1.0)  # Default weight if missing from idf map somehow
            
    return gain


def rank_groups(groups: List[CoverageGroup]) -> List[CoverageGroup]:
    """Rank groups using a greedy coverage algorithm."""
    if not groups:
        return []
        
    idf_map = compute_group_idf(groups)
    covered: Set[Tuple[str, str]] = set()
    ranked: List[CoverageGroup] = []
    
    remaining = list(groups)
    
    while remaining:
        best_idx = -1
        best_gain = -1.0
        best_count = -1
        best_num = float('inf')
        
        for i, group in enumerate(remaining):
            gain = marginal_gain(group, covered, idf_map)
            count = len(flatten_features(group))
            num = group.representative_report_num
            
            # Tie-breaker:
            # 1. `marginal_gain` 更大
            # 2. `coverage_feature_count` 更大
            # 3. `representative_report_num` 更小
            
            if gain > best_gain:
                best_idx = i
                best_gain = gain
                best_count = count
                best_num = num
            elif gain == best_gain:
                if count > best_count:
                    best_idx = i
                    best_gain = gain
                    best_count = count
                    best_num = num
                elif count == best_count:
                    if num < best_num:
                        best_idx = i
                        best_gain = gain
                        best_count = count
                        best_num = num
                        
        best_group = remaining.pop(best_idx)
        ranked.append(best_group)
        
        # update covered features
        best_features = flatten_features(best_group)
        covered.update(best_features)
        
    return ranked