import pytest
from transcribe.core.schema import CandidateAnnotation, CandidateList

def test_candidate_schema():
    cand = CandidateAnnotation(cell_type="T Cell", confidence="high", reasoning="Has CD3E")
    assert cand.cell_type == "T Cell"
    assert cand.confidence == "high"
    
def test_candidate_list_schema():
    cand1 = CandidateAnnotation(cell_type="CD4+ T Cell", confidence="high", reasoning="...")
    cand2 = CandidateAnnotation(cell_type="CD8+ T Cell", confidence="medium", reasoning="...")
    
    clist = CandidateList(candidates=[cand1, cand2])
    assert len(clist.candidates) == 2
    assert clist.candidates[0].cell_type == "CD4+ T Cell"
