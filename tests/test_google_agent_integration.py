"""
Test Google Agent Integration
"""

import pytest
from unittest.mock import Mock, patch
from utils.google_agent_integration import GoogleAgentIntegration, GoogleAgentModelInterface
from utils.config_manager import ConfigManager


class TestGoogleAgentIntegration:
    """Test cases for Google Agent Integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_config = Mock(spec=ConfigManager)
        self.mock_config.get_all_api_keys.return_value = {
            "serpapi": "test_serpapi_key",
            "tavily": "test_tavily_key",
            "google": "test_google_key"
        }
        self.integration = GoogleAgentIntegration(self.mock_config)
    
    def test_initialization(self):
        """Test GoogleAgentIntegration initialization"""
        assert self.integration.config_manager == self.mock_config
        assert "serpapi" in self.integration.search_providers
        assert "tavily" in self.integration.search_providers
    
    def test_get_server_time(self):
        """Test server time generation"""
        time_str = self.integration.get_server_time()
        assert isinstance(time_str, str)
        assert "UTC" in time_str
    
    def test_search_web_fallback(self):
        """Test fallback search functionality"""
        results = self.integration.search_web_fallback("test query", 3)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all("title" in result for result in results)
        assert all("url" in result for result in results)
        assert all("snippet" in result for result in results)
    
    def test_format_search_results_for_context(self):
        """Test search results formatting"""
        sample_results = [
            {
                "title": "Test Title",
                "url": "https://example.com",
                "snippet": "Test snippet",
                "source": "example.com"
            }
        ]
        
        formatted = self.integration.format_search_results_for_context(sample_results)
        assert "## Web Search Results:" in formatted
        assert "Test Title" in formatted
        assert "https://example.com" in formatted
        assert "Test snippet" in formatted
    
    def test_extract_citations_from_results(self):
        """Test citation extraction"""
        sample_results = [
            {"url": "https://example1.com"},
            {"url": "https://example2.com"},
            {"url": "https://example1.com"}  # Duplicate
        ]
        
        citations = self.integration.extract_citations_from_results(sample_results)
        assert len(citations) == 2  # No duplicates
        assert "https://example1.com" in citations
        assert "https://example2.com" in citations
    
    def test_validate_configuration(self):
        """Test configuration validation"""
        validation = self.integration.validate_configuration()
        assert isinstance(validation, dict)
        assert "basic_functionality" in validation
        assert validation["basic_functionality"] is True
        assert "serpapi_configured" in validation
        assert "tavily_configured" in validation
    
    @patch('trafilatura.fetch_url')
    @patch('trafilatura.extract')
    def test_get_website_content_from_url(self, mock_extract, mock_fetch):
        """Test website content extraction"""
        mock_fetch.return_value = "mock_html_content"
        mock_extract.return_value = '{"title": "Test Title", "raw_text": "Test content"}'
        
        result = self.integration.get_website_content_from_url("https://example.com")
        
        assert "Test Title" in result
        assert "Test content" in result
        assert "https://example.com" in result
        
        mock_fetch.assert_called_once_with("https://example.com")
        mock_extract.assert_called_once()


class TestGoogleAgentModelInterface:
    """Test cases for Google Agent Model Interface"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_config = Mock(spec=ConfigManager)
        self.model_interface = GoogleAgentModelInterface(self.mock_config)
    
    def test_initialization(self):
        """Test GoogleAgentModelInterface initialization"""
        assert self.model_interface.config_manager == self.mock_config
    
    def test_generate_response(self):
        """Test response generation"""
        query = "test query"
        context = "test search context"
        
        response = self.model_interface.generate_response(
            query=query,
            search_context=context,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1024
        )
        
        assert isinstance(response, str)
        assert query in response
        assert context in response
        assert "gpt-4o" in response


class TestIntegrationFlow:
    """Test complete integration flow"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.mock_config = Mock(spec=ConfigManager)
        self.mock_config.get_all_api_keys.return_value = {}
        self.integration = GoogleAgentIntegration(self.mock_config)
        self.model_interface = GoogleAgentModelInterface(self.mock_config)
    
    def test_complete_search_flow(self):
        """Test complete search and response flow"""
        query = "test search query"
        
        # Perform search (will use fallback)
        results = self.integration.search_web(query, 3)
        assert len(results) >= 1
        
        # Format results
        context = self.integration.format_search_results_for_context(results)
        assert isinstance(context, str)
        
        # Generate response
        response = self.model_interface.generate_response(
            query=query,
            search_context=context
        )
        assert isinstance(response, str)
        assert query in response
        
        # Extract citations
        citations = self.integration.extract_citations_from_results(results)
        assert isinstance(citations, list)


if __name__ == "__main__":
    pytest.main([__file__])
