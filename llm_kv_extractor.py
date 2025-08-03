# LLM-Based Key-Value Extractor
# Fallback extractor using Large Language Models for robust document understanding

import json
import re
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"

@dataclass
class LLMKeyValuePair:
    """LLM-extracted key-value pair with confidence and metadata"""
    key: str
    value: str
    confidence: float = 0.8  # LLM extractions generally high confidence
    extraction_method: str = "llm_extraction"
    llm_provider: str = "unknown"
    response_metadata: Dict = None

class LLMKeyValueExtractor:
    """
    LLM-based key-value extractor for robust document understanding
    Supports multiple LLM providers with graceful fallbacks
    """
    
    def __init__(self, 
                 primary_provider: LLMProvider = LLMProvider.OPENAI,
                 fallback_providers: List[LLMProvider] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or [LLMProvider.OLLAMA]
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize available providers
        self.available_providers = {}
        self._initialize_providers()
        
        # Extraction statistics for learning
        self.extraction_stats = {
            "total_requests": 0,
            "successful_extractions": 0,
            "provider_usage": {},
            "average_pairs_extracted": 0.0
        }
    
    def _initialize_providers(self):
        """Initialize available LLM providers"""
        
        # OpenAI
        try:
            if os.getenv("OPENAI_API_KEY"):
                import openai
                self.available_providers[LLMProvider.OPENAI] = {
                    "client": openai.OpenAI(),
                    "model": "gpt-4o-mini",  # Cost-effective for extraction
                    "backup_model": "gpt-3.5-turbo"
                }
                logger.info("OpenAI provider initialized")
        except ImportError:
            logger.debug("OpenAI not available - install: pip install openai")
        except Exception as e:
            logger.debug(f"OpenAI initialization failed: {e}")
        
        # Anthropic Claude
        try:
            if os.getenv("ANTHROPIC_API_KEY"):
                import anthropic
                self.available_providers[LLMProvider.ANTHROPIC] = {
                    "client": anthropic.Anthropic(),
                    "model": "claude-3-haiku-20240307",  # Fast and cost-effective
                    "backup_model": "claude-3-sonnet-20240229"
                }
                logger.info("Anthropic provider initialized")
        except ImportError:
            logger.debug("Anthropic not available - install: pip install anthropic")
        except Exception as e:
            logger.debug(f"Anthropic initialization failed: {e}")
        
        # Ollama (local)
        try:
            import requests
            # Test if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                # Get available models
                models_data = response.json()
                available_models = [model['name'] for model in models_data.get('models', [])]
                
                # Choose best available model
                preferred_models = ["llama3.1:8b", "llama3:8b", "llama2:7b", "mistral:7b", "codellama:7b"]
                selected_model = None
                backup_model = None
                
                for model in preferred_models:
                    if model in available_models:
                        if selected_model is None:
                            selected_model = model
                        elif backup_model is None:
                            backup_model = model
                            break
                
                if selected_model:
                    config = {
                        "base_url": "http://localhost:11434",
                        "model": selected_model,
                        "available_models": available_models
                    }
                    if backup_model:
                        config["backup_model"] = backup_model
                    
                    self.available_providers[LLMProvider.OLLAMA] = config
                    logger.info(f"Ollama provider initialized with model: {selected_model}")
                else:
                    logger.warning("Ollama is running but no suitable models found. Available models: " + str(available_models))
            else:
                logger.debug(f"Ollama not available: HTTP {response.status_code}")
        except Exception as e:
            logger.debug(f"Ollama initialization failed: {e}")
        
        # Azure OpenAI
        try:
            if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
                import openai
                self.available_providers[LLMProvider.AZURE_OPENAI] = {
                    "client": openai.AzureOpenAI(
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version="2024-02-01",
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                    ),
                    "model": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                    "backup_model": "gpt-35-turbo"
                }
                logger.info("Azure OpenAI provider initialized")
        except Exception as e:
            logger.debug(f"Azure OpenAI initialization failed: {e}")
        
        if not self.available_providers:
            logger.warning("No LLM providers available! Set API keys or run Ollama locally.")
    
    def build_extraction_prompt(self, text: str, document_type: str = "document") -> str:
        """
        Build optimized prompt for key-value extraction
        """
        prompt = f"""You are a document understanding assistant specialized in extracting structured information.

TASK: Extract ALL key-value pairs from the text below. Focus on:
- Names, IDs, numbers, dates, addresses
- Form fields, labels, and their corresponding values  
- Document metadata (issue dates, validity, etc.)
- Contact information (phone, email, etc.)

RULES:
1. Return ONLY valid JSON format: {{"key": "value", "key2": "value2"}}
2. Use clear, descriptive keys (e.g., "Full Name" not just "Name")
3. Preserve original values exactly as written
4. Skip empty, unclear, or duplicate information
5. For dates, keep original format
6. If uncertain about a pairing, include it with a clear key

DOCUMENT TYPE: {document_type}

TEXT TO ANALYZE:
{text}

JSON OUTPUT:"""
        
        return prompt
    
    def extract_key_value_pairs(self, text: str, document_type: str = "document") -> List[LLMKeyValuePair]:
        """
        Extract key-value pairs using LLM with provider fallback
        """
        if not text.strip():
            return []
        
        self.extraction_stats["total_requests"] += 1
        
        # Try providers in order of preference
        providers_to_try = [self.primary_provider] + self.fallback_providers
        
        for provider in providers_to_try:
            if provider not in self.available_providers:
                continue
            
            try:
                logger.info(f"Attempting LLM extraction with {provider.value}")
                
                # Extract using this provider
                result = self._extract_with_provider(provider, text, document_type)
                
                if result:
                    self.extraction_stats["successful_extractions"] += 1
                    self.extraction_stats["provider_usage"][provider.value] = \
                        self.extraction_stats["provider_usage"].get(provider.value, 0) + 1
                    
                    self.extraction_stats["average_pairs_extracted"] = \
                        (self.extraction_stats["average_pairs_extracted"] * 
                         (self.extraction_stats["successful_extractions"] - 1) + len(result)) / \
                        self.extraction_stats["successful_extractions"]
                    
                    logger.info(f"LLM extraction successful with {provider.value}: {len(result)} pairs")
                    return result
                    
            except Exception as e:
                logger.warning(f"LLM extraction failed with {provider.value}: {e}")
                continue
        
        logger.error("All LLM providers failed for extraction")
        return []
    
    def _extract_with_provider(self, provider: LLMProvider, text: str, document_type: str) -> List[LLMKeyValuePair]:
        """Extract using specific provider"""
        
        prompt = self.build_extraction_prompt(text, document_type)
        provider_config = self.available_providers[provider]
        
        if provider == LLMProvider.OPENAI:
            return self._extract_openai(provider_config, prompt, provider.value)
        
        elif provider == LLMProvider.ANTHROPIC:
            return self._extract_anthropic(provider_config, prompt, provider.value)
        
        elif provider == LLMProvider.OLLAMA:
            return self._extract_ollama(provider_config, prompt, provider.value)
        
        elif provider == LLMProvider.AZURE_OPENAI:
            return self._extract_azure_openai(provider_config, prompt, provider.value)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _extract_openai(self, config: Dict, prompt: str, provider_name: str) -> List[LLMKeyValuePair]:
        """Extract using OpenAI GPT"""
        try:
            response = config["client"].chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": "You are a precise document analysis assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content, provider_name, {
                "model": config["model"],
                "tokens_used": response.usage.total_tokens if response.usage else 0
            })
            
        except Exception as e:
            # Try backup model
            if "backup_model" in config:
                logger.info(f"Trying OpenAI backup model: {config['backup_model']}")
                response = config["client"].chat.completions.create(
                    model=config["backup_model"],
                    messages=[
                        {"role": "system", "content": "You are a document analysis assistant. Return only JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                content = response.choices[0].message.content
                return self._parse_llm_response(content, provider_name, {"model": config["backup_model"]})
            else:
                raise e
    
    def _extract_anthropic(self, config: Dict, prompt: str, provider_name: str) -> List[LLMKeyValuePair]:
        """Extract using Anthropic Claude"""
        try:
            response = config["client"].messages.create(
                model=config["model"],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            return self._parse_llm_response(content, provider_name, {
                "model": config["model"],
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens
            })
            
        except Exception as e:
            # Try backup model
            if "backup_model" in config:
                logger.info(f"Trying Anthropic backup model: {config['backup_model']}")
                response = config["client"].messages.create(
                    model=config["backup_model"],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text
                return self._parse_llm_response(content, provider_name, {"model": config["backup_model"]})
            else:
                raise e
    
    def _extract_ollama(self, config: Dict, prompt: str, provider_name: str) -> List[LLMKeyValuePair]:
        """Extract using Ollama (local)"""
        import requests
        
        try:
            response = requests.post(
                f"{config['base_url']}/api/generate",
                json={
                    "model": config["model"],
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                content = response.json()["response"]
                return self._parse_llm_response(content, provider_name, {"model": config["model"]})
            else:
                raise Exception(f"Ollama request failed: {response.status_code}")
                
        except Exception as e:
            # Try backup model
            if "backup_model" in config:
                logger.info(f"Trying Ollama backup model: {config['backup_model']}")
                response = requests.post(
                    f"{config['base_url']}/api/generate",
                    json={
                        "model": config["backup_model"],
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": self.temperature}
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    content = response.json()["response"]
                    return self._parse_llm_response(content, provider_name, {"model": config["backup_model"]})
            raise e
    
    def _extract_azure_openai(self, config: Dict, prompt: str, provider_name: str) -> List[LLMKeyValuePair]:
        """Extract using Azure OpenAI"""
        # Similar to OpenAI but with Azure client
        return self._extract_openai(config, prompt, provider_name)
    
    def _parse_llm_response(self, content: str, provider_name: str, metadata: Dict) -> List[LLMKeyValuePair]:
        """
        Parse LLM response and convert to LLMKeyValuePair objects
        """
        try:
            # Clean the response - remove markdown formatting if present
            content = content.strip()
            
            # Remove markdown code blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            # Parse JSON
            kv_dict = json.loads(content)
            
            if not isinstance(kv_dict, dict):
                logger.warning(f"LLM response is not a dictionary: {type(kv_dict)}")
                return []
            
            # Convert to LLMKeyValuePair objects
            pairs = []
            for key, value in kv_dict.items():
                if key and value and str(value).strip():  # Skip empty values
                    pairs.append(LLMKeyValuePair(
                        key=str(key).strip(),
                        value=str(value).strip(),
                        confidence=0.8,  # LLM extractions generally reliable
                        extraction_method="llm_extraction",
                        llm_provider=provider_name,
                        response_metadata=metadata
                    ))
            
            logger.info(f"Successfully parsed {len(pairs)} key-value pairs from LLM response")
            return pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw content: {content}")
            
            # Fallback: try to extract key-value pairs with regex
            return self._fallback_parse_response(content, provider_name, metadata)
        
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _fallback_parse_response(self, content: str, provider_name: str, metadata: Dict) -> List[LLMKeyValuePair]:
        """
        Fallback parsing when JSON parsing fails
        """
        pairs = []
        
        try:
            # Look for key-value patterns in the text
            patterns = [
                r'"([^"]+)":\s*"([^"]+)"',  # "key": "value"
                r'(\w+(?:\s+\w+)*?):\s*(.+?)(?:\n|$)',  # key: value
                r'([A-Za-z\s]+?)[-–]\s*(.+?)(?:\n|$)',  # key - value
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for key, value in matches:
                    key = key.strip()
                    value = value.strip()
                    
                    if len(key) > 2 and len(value) > 0 and len(key) < 50:
                        pairs.append(LLMKeyValuePair(
                            key=key,
                            value=value,
                            confidence=0.6,  # Lower confidence for regex parsing
                            extraction_method="llm_fallback_regex",
                            llm_provider=provider_name,
                            response_metadata=metadata
                        ))
            
            # Remove duplicates
            seen = set()
            unique_pairs = []
            for pair in pairs:
                pair_key = (pair.key.lower(), pair.value.lower())
                if pair_key not in seen:
                    seen.add(pair_key)
                    unique_pairs.append(pair)
            
            logger.info(f"Fallback parsing extracted {len(unique_pairs)} pairs")
            return unique_pairs[:20]  # Limit to prevent noise
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return []
    
    def get_extraction_statistics(self) -> Dict:
        """Get statistics about LLM extraction performance"""
        return {
            "total_requests": self.extraction_stats["total_requests"],
            "successful_extractions": self.extraction_stats["successful_extractions"],
            "success_rate": self.extraction_stats["successful_extractions"] / max(1, self.extraction_stats["total_requests"]),
            "provider_usage": self.extraction_stats["provider_usage"],
            "average_pairs_per_extraction": self.extraction_stats["average_pairs_extracted"],
            "available_providers": list(self.available_providers.keys()),
            "primary_provider": self.primary_provider.value
        }
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available"""
        return len(self.available_providers) > 0
    
    def get_provider_info(self) -> Dict:
        """Get information about available providers"""
        info = {}
        for provider, config in self.available_providers.items():
            info[provider.value] = {
                "model": config.get("model", "unknown"),
                "backup_model": config.get("backup_model", "none"),
                "status": "available"
            }
        
        return info

# Convenience function for quick testing
def test_llm_extraction(text: str, provider: LLMProvider = LLMProvider.OPENAI) -> Dict:
    """
    Quick test function for LLM extraction
    """
    extractor = LLMKeyValueExtractor(primary_provider=provider)
    
    if not extractor.is_available():
        return {"error": "No LLM providers available"}
    
    pairs = extractor.extract_key_value_pairs(text)
    
    return {
        "pairs_extracted": len(pairs),
        "pairs": [{"key": p.key, "value": p.value, "confidence": p.confidence} for p in pairs],
        "provider_used": pairs[0].llm_provider if pairs else "none",
        "statistics": extractor.get_extraction_statistics()
    }

if __name__ == "__main__":
    # Quick test
    test_text = """
    Registration Number: MH12AB1234
    Owner Name: Divija Kalluri  
    PUC Valid Till: 15-Aug-2025
    Vehicle Type: Two Wheeler
    Engine Number: ABC123XYZ
    """
    
    result = test_llm_extraction(test_text)
    print(json.dumps(result, indent=2))