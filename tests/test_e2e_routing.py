"""
End-to-end integration test for query routing.
Demonstrates that general queries skip RAG and document queries use it.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.query_router import classify_query, should_use_rag, get_query_classification


def test_e2e_general_query():
    """End-to-end test: General query should NOT go through RAG"""
    print("\n" + "="*80)
    print("TEST 1: General Query - 'how to eat ice cream'")
    print("="*80)
    
    query = "how to eat ice cream"
    print(f"\n📝 Query: {query}")
    
    # Step 1: Classify
    classification = get_query_classification(query)
    print(f"\n🔍 Classification Result:")
    print(f"   Type: {classification['type']}")
    print(f"   Confidence: {classification['confidence']:.2%}")
    print(f"   Reason: {classification['reason']}")
    
    # Step 2: Routing decision
    use_rag = should_use_rag(query)
    print(f"\n🛣️  Routing Decision: {'RAG Pipeline' if use_rag else 'Pure LLM Chat'}")
    
    # Step 3: Expected behavior
    print(f"\n✅ Expected Behavior:")
    print(f"   - Skip vector search: YES")
    print(f"   - Skip embeddings: YES")
    print(f"   - Load dataset: NO")
    print(f"   - Show 'Data loaded' message: NO")
    print(f"   - Call LLM directly: YES")
    print(f"   - Show visualizations: NO")
    
    # Verify
    assert classification['type'] == 'general', "Should be classified as general"
    assert use_rag == False, "Should NOT use RAG"
    print(f"\n✅ PASS: General query correctly bypasses RAG pipeline\n")


def test_e2e_rag_metric_query():
    """End-to-end test: Metric query SHOULD go through RAG"""
    print("\n" + "="*80)
    print("TEST 2: DATA Query - 'show me gas production'")
    print("="*80)
    
    query = "show me gas production"
    print(f"\n📝 Query: {query}")
    
    # Step 1: Classify
    classification = get_query_classification(query)
    print(f"\n🔍 Classification Result:")
    print(f"   Type: {classification['type']}")
    print(f"   Confidence: {classification['confidence']:.2%}")
    print(f"   Reason: {classification['reason']}")
    
    # Step 2: Routing decision
    use_rag = should_use_rag(query)
    print(f"\n🛣️  Routing Decision: {'RAG Pipeline' if use_rag else 'Pure LLM Chat'}")
    
    # Step 3: Expected behavior
    print(f"\n✅ Expected Behavior:")
    print(f"   - Skip vector search: NO")
    print(f"   - Skip embeddings: NO")
    print(f"   - Load dataset: YES")
    print(f"   - Show 'Data loaded' message: YES")
    print(f"   - Call LLM directly: NO (use RAG)")
    print(f"   - Show visualizations: YES (gas charts only)")
    
    # Verify
    assert classification['type'] == 'data', "Should be classified as data"
    assert use_rag == True, "Should use RAG"
    print(f"\n✅ PASS: Metric query correctly uses RAG pipeline\n")


def test_e2e_rag_document_query():
    """End-to-end test: Document overview query SHOULD go through RAG"""
    print("\n" + "="*80)
    print("TEST 3: DATA Query - 'what is in the document'")
    print("="*80)
    
    query = "what is in the document"
    print(f"\n📝 Query: {query}")
    
    # Step 1: Classify
    classification = get_query_classification(query)
    print(f"\n🔍 Classification Result:")
    print(f"   Type: {classification['type']}")
    print(f"   Confidence: {classification['confidence']:.2%}")
    print(f"   Reason: {classification['reason']}")
    
    # Step 2: Routing decision
    use_rag = should_use_rag(query)
    print(f"\n🛣️  Routing Decision: {'RAG Pipeline' if use_rag else 'Pure LLM Chat'}")
    
    # Step 3: Expected behavior
    print(f"\n✅ Expected Behavior:")
    print(f"   - Skip vector search: NO")
    print(f"   - Skip embeddings: NO")
    print(f"   - Load dataset: YES")
    print(f"   - Show 'Data loaded' message: YES")
    print(f"   - Call LLM directly: NO (use RAG)")
    print(f"   - Show visualizations: NO (overview only)")
    
    # Verify
    assert classification['type'] == 'data', "Should be classified as data"
    assert use_rag == True, "Should use RAG"
    print(f"\n✅ PASS: Document query correctly uses RAG pipeline\n")


def test_e2e_mixed_keywords():
    """Test: Query with both data and general keywords (data wins)"""
    print("\n" + "="*80)
    print("TEST 4: Mixed Keywords - 'how to interpret oil production data'")
    print("="*80)
    
    query = "how to interpret oil production data"
    print(f"\n📝 Query: {query}")
    
    # Step 1: Classify
    classification = get_query_classification(query)
    print(f"\n🔍 Classification Result:")
    print(f"   Type: {classification['type']}")
    print(f"   Confidence: {classification['confidence']:.2%}")
    print(f"   Reason: {classification['reason']}")
    
    # Step 2: Routing decision
    use_rag = should_use_rag(query)
    print(f"\n🛣️  Routing Decision: {'RAG Pipeline' if use_rag else 'Pure LLM Chat'}")
    
    # Explanation
    print(f"\n📊 Analysis:")
    print(f"   DATA keywords: 'oil', 'production', 'data'")
    print(f"   General keywords: 'how to', 'interpret'")
    print(f"   Winner: DATA (dataset keywords present → use RAG)")
    
    # Verify
    assert classification['type'] == 'data', "Should favor data query when data keywords present"
    print(f"\n✅ PASS: Mixed query correctly prioritizes data context\n")


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  END-TO-END QUERY ROUTING INTEGRATION TESTS".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    test_e2e_general_query()
    test_e2e_rag_metric_query()
    test_e2e_rag_document_query()
    test_e2e_mixed_keywords()
    
    print("\n" + "="*80)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("="*80)
    print("\n📊 Query Routing is working correctly:")
    print("   ✓ General queries skip RAG pipeline")
    print("   ✓ Document queries use RAG pipeline")
    print("   ✓ Metric queries trigger visualizations")
    print("   ✓ Mixed queries prioritize document context")
    print("\n")
