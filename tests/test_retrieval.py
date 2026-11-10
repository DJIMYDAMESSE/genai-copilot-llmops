from copilot.ingest import chunk_markdown
from copilot.retriever import HybridRetriever, tokenize


def test_chunking_by_section():
    md = "# Titre\n\nIntro assez longue pour être conservée comme chunk.\n\n## Section A\n\n- point un détaillé ici\n- point deux détaillé ici\n"
    chunks = chunk_markdown("doc", md)
    assert [c.section for c in chunks] == ["", "Section A"]
    assert "•" in chunks[1].text  # les puces sont conservées
    assert chunks[1].source == "Titre › Section A"


def test_long_section_is_split_with_overlap():
    md = "# T\n\n## S\n\n" + " ".join(f"Phrase numéro {i} avec du contenu." for i in range(80))
    chunks = chunk_markdown("doc", md, size=300, overlap=50)
    assert len(chunks) > 3 and all(len(c.text) <= 360 for c in chunks)


def test_tokenize_normalizes_accents_and_stopwords():
    assert tokenize("Les délais de résolution") == ["delai", "resolution"]


def test_hybrid_search_finds_right_document(retriever):
    hits = retriever.search("délai de prise en charge incident P1", k=3)
    assert hits[0].chunk.doc_id == "02_procedure_incident"
    assert 0 < hits[0].relevance <= 1


def test_out_of_scope_has_low_relevance(retriever, settings):
    hits = retriever.search("capitale de l'Australie", k=3)
    assert hits[0].relevance < settings.min_relevance


def test_index_roundtrip(retriever, tmp_path):
    p = tmp_path / "idx.pkl"
    retriever.save(p)
    loaded = HybridRetriever.load(p)
    assert len(loaded.chunks) == len(retriever.chunks)
    assert loaded.search("mot de passe", k=1)[0].chunk.doc_id == "01_politique_securite_si"
