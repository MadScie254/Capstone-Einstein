"""
Einstein - Electricity Theft Detection System
Test Suite - API Tests

Author: Capstone Team Einstein
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import json

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'backend'))

from app.main import app
from app.models import create_dummy_models

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_models():
    """Ensure dummy models exist for testing."""
    artifacts_dir = str(Path(__file__).parent.parent / 'artifacts')
    create_dummy_models(artifacts_dir)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test basic health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestScoreEndpoint:
    """Test scoring endpoint."""
    
    def test_score_valid_input(self):
        """Test scoring with valid consumption data."""
        payload = {
            "consumption": [100.0, 95.0, 105.0, 98.0, 92.0, 88.0, 110.0, 
                          107.0, 45.0, 48.0, 95.0, 97.0],
            "customer_id": "TEST_001"
        }
        
        response = client.post("/score", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "probability" in data
        assert "risk_level" in data
        assert "customer_id" in data
        assert 0.0 <= data["probability"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_score_minimal_input(self):
        """Test scoring with minimal data."""
        payload = {
            "consumption": [100.0, 95.0, 90.0]
        }
        
        response = client.post("/score", json=payload)
        
        # Should succeed even with minimal data
        assert response.status_code == 200
    
    def test_score_with_zeros(self):
        """Test scoring with zero consumption values."""
        payload = {
            "consumption": [100.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 100.0, 95.0]
        }
        
        response = client.post("/score", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # With many zeros, should likely be flagged as risky
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_score_empty_consumption(self):
        """Test scoring with empty consumption list."""
        payload = {
            "consumption": []
        }
        
        response = client.post("/score", json=payload)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_score_invalid_type(self):
        """Test scoring with invalid consumption type."""
        payload = {
            "consumption": "not a list"
        }
        
        response = client.post("/score", json=payload)
        
        assert response.status_code == 422


class TestExplainEndpoint:
    """Test explanation endpoint."""
    
    def test_explain_valid_input(self):
        """Test explanation with valid data."""
        payload = {
            "consumption": [100.0, 95.0, 105.0, 98.0, 92.0, 88.0, 110.0, 
                          107.0, 45.0, 48.0, 95.0, 97.0]
        }
        
        response = client.post("/explain", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "probability" in data
        assert "top_features" in data
        assert "consumption_stats" in data


class TestClustersEndpoint:
    """Test clusters endpoint."""
    
    def test_get_clusters(self):
        """Test getting cluster profiles."""
        response = client.get("/clusters")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "n_clusters" in data or "profiles" in data


class TestMetricsEndpoint:
    """Test metrics endpoint."""
    
    def test_get_metrics(self):
        """Test getting model metrics."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Should return some metrics or placeholder


class TestSyntheticEndpoint:
    """Test synthetic data generation endpoint."""
    
    def test_generate_synthetic_default(self):
        """Test synthetic data generation with defaults."""
        response = client.post("/synthetic")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_samples" in data
        assert data["total_samples"] == 100  # Default
    
    def test_generate_synthetic_custom(self):
        """Test synthetic data generation with custom params."""
        response = client.post("/synthetic?n_samples=50&theft_ratio=0.3")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_samples"] == 50


class TestBatchScoring:
    """Test batch scoring endpoint."""
    
    def test_batch_score_multiple(self):
        """Test batch scoring multiple customers."""
        payload = {
            "customers": [
                {"consumption": [100.0, 95.0, 105.0, 98.0, 92.0], "customer_id": "C1"},
                {"consumption": [50.0, 0.0, 0.0, 25.0, 30.0], "customer_id": "C2"}
            ]
        }
        
        response = client.post("/score/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "total_processed" in data
        assert data["total_processed"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
