"""
Unit tests for query classification and routing.
Tests whether queries are correctly classified as RAG or general LLM queries.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.query_router import classify_query, should_use_rag, get_query_classification


def test_general_query_ice_cream():
    """Test: General knowledge question about ice cream"""
    query = "how to eat ice cream"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "general", f"Expected 'general' but got '{classification}'"
    print("✅ PASS: General query correctly identified\n")


def test_metric_query_gas_production():
    """Test: Specific metric query about gas production"""
    query = "show me gas production"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "data", f"Expected 'data' but got '{classification}'"
    assert should_use_rag(query) == True, "Should route to RAG"
    print("✅ PASS: Metric query correctly identified\n")


def test_document_summary_query():
    """Test: Document summary request"""
    query = "what is in the document"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "data", f"Expected 'data' but got '{classification}'"
    assert should_use_rag(query) == True, "Should route to RAG"
    print("✅ PASS: Document query correctly identified\n")


def test_detailed_analysis_query():
    """Test: Detailed document analysis request"""
    query = "give me detailed document analysis"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "data", f"Expected 'data' but got '{classification}'"
    print("✅ PASS: Detailed analysis query correctly identified\n")


def test_oil_production_query():
    """Test: Specific oil production metric"""
    query = "what is total oil production"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "data", f"Expected 'data' but got '{classification}'"
    print("✅ PASS: Oil production query correctly identified\n")


def test_water_injection_query():
    """Test: Water injection metrics"""
    query = "show water injection statistics"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "data", f"Expected 'data' but got '{classification}'"
    print("✅ PASS: Water injection query correctly identified\n")


def test_general_math_query():
    """Test: General math/calculation question"""
    query = "what is 2 plus 2"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "general", f"Expected 'general' but got '{classification}'"
    print("✅ PASS: Math query correctly identified\n")


def test_general_how_to_query():
    """Test: General 'how to' question"""
    query = "how to cook pasta"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "general", f"Expected 'general' but got '{classification}'"
    print("✅ PASS: How-to query correctly identified\n")


def test_data_quality_query():
    """Test: Data quality analysis"""
    query = "analyze data quality and missing values"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "data", f"Expected 'data' but got '{classification}'"
    print("✅ PASS: Data quality query correctly identified\n")


def test_chart_visualization_query():
    """Test: Chart/visualization request"""
    query = "show me a chart of production trends"
    classification, confidence = classify_query(query)
    
    print(f"Query: '{query}'")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Should use RAG: {should_use_rag(query)}")
    
    assert classification == "data", f"Expected 'data' but got '{classification}'"
    print("✅ PASS: Chart visualization query correctly identified\n")


def test_get_classification_info():
    """Test: Get detailed classification information"""
    query = "how to eat ice cream"
    info = get_query_classification(query)
    
    print(f"Query: '{query}'")
    print(f"Classification Info: {info}")
    
    assert info["type"] == "general", "Should be general"
    assert info["use_rag"] == False, "Should not use RAG"
    print("✅ PASS: Classification info correctly returned\n")


if __name__ == "__main__":
    print("=" * 80)
    print("QUERY CLASSIFICATION TESTS")
    print("=" * 80)
    print()
    
    # Test general queries (should NOT use RAG)
    print("GENERAL QUERIES (should NOT route to RAG):")
    print("-" * 80)
    test_general_query_ice_cream()
    test_general_math_query()
    test_general_how_to_query()
    
    # Test RAG/document queries (should use RAG)
    print("DOCUMENT/RAG QUERIES (should route to RAG):")
    print("-" * 80)
    test_metric_query_gas_production()
    test_document_summary_query()
    test_detailed_analysis_query()
    test_oil_production_query()
    test_water_injection_query()
    test_data_quality_query()
    test_chart_visualization_query()
    
    # Test utility functions
    print("UTILITY FUNCTIONS:")
    print("-" * 80)
    test_get_classification_info()
    
    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
