import pytest
from transcribe.core.schema import CandidateAnnotation, CandidateList

def test_candidate_schema():
    cand = CandidateAnnotation(cell_type="T Cell", confidence=0.9, reasoning="Has CD3E")
    assert cand.cell_type == "T Cell"
    assert cand.confidence == 0.9
    
def test_candidate_list_schema():
    cand1 = CandidateAnnotation(cell_type="CD4+ T Cell", confidence=0.8, reasoning="...")
    cand2 = CandidateAnnotation(cell_type="CD8+ T Cell", confidence=0.7, reasoning="...")
    
    clist = CandidateList(candidates=[cand1, cand2])
    assert len(clist.candidates) == 2
    assert clist.candidates[0].cell_type == "CD4+ T Cell"
