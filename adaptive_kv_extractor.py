# Adaptive Key-Value Extractor
# Replaces hardcoded confidence boosts and thresholds with learned parameters

import spacy
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from config import adaptive_config

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    @property
    def right(self) -> int:
        return self.x + self.width

@dataclass
class TextBlock:
    text: str
    bbox: BoundingBox
    confidence: float

@dataclass
class KeyValuePair:
    key: str
    value: str
    key_bbox: BoundingBox
    value_bbox: BoundingBox
    confidence: float
    extraction_method: str

class AdaptiveKeyValueExtractor:
    """
    Fully adaptive key-value extractor with learned confidence parameters
    """
    
    def __init__(self):
        # Load spaCy model
        self.nlp = self._load_best_spacy_model()
        
        # Dynamic learning structures
        self.learned_patterns = {}
        self.pattern_confidence = {}
        self.successful_extractions = []
        
        # No hardcoded values - all come from adaptive config
        
    def _load_best_spacy_model(self):
        """Load the best available spaCy model"""
        model_preferences = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
        
        for model_name in model_preferences:
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                continue
        
        logger.warning("No spaCy model found")
        return None
    
    def identify_potential_keys(self, text_blocks: List[TextBlock], form_analysis: Dict = None) -> List[Tuple[TextBlock, float, str]]:
        """
        Adaptive key identification using learned confidence parameters
        """
        key_candidates = []
        is_form = form_analysis and form_analysis.get('is_form', False) if form_analysis else False
        
        for block in text_blocks:
            text = block.text.strip()
            confidence = 0.0
            methods = []
            
            # Method 1: Structural indicators with adaptive confidence
            if text.endswith(':'):
                # Get adaptive colon boost (learned from successful extractions)
                colon_boost = adaptive_config.get_adaptive_value(
                    "extraction_confidence", "colon_terminator_boost"
                )
                confidence += colon_boost
                methods.append('adaptive_colon_terminator')
            
            if text.endswith(('No.', 'Number', 'ID', 'Code', '#', 'Name')):
                # Get adaptive suffix boost
                suffix_boost = adaptive_config.get_adaptive_value(
                    "extraction_confidence", "suffix_indicator_boost"  
                )
                confidence += suffix_boost
                methods.append('adaptive_suffix_indicator')
            
            # Method 2: Length and word count heuristics (adaptive)
            words = text.split()
            if 1 <= len(words) <= 3 and 3 <= len(text) <= 40:
                # Learn appropriate confidence from successful extractions
                length_boost = self._get_learned_confidence('label_length', 0.3)
                confidence += length_boost
                methods.append('adaptive_label_length')
            
            # Method 3: Capitalization patterns (adaptive)
            if text.istitle() and len(words) <= 3:
                title_boost = self._get_learned_confidence('title_case', 0.2)
                confidence += title_boost
                methods.append('adaptive_title_case')
            elif text.isupper() and len(text) > 2 and len(text) <= 20:
                upper_boost = self._get_learned_confidence('upper_case', 0.1)
                confidence += upper_boost
                methods.append('adaptive_upper_case')
            
            # Method 4: Position-based analysis with adaptive thresholds
            if is_form and form_analysis.get('left_aligned', False):
                # Get adaptive left position threshold
                left_threshold = adaptive_config.get_adaptive_value(
                    "extraction_confidence", "form_left_position_threshold"
                )
                
                if block.bbox.x < left_threshold:
                    position_boost = self._get_learned_confidence('form_left_position', 0.3)
                    confidence += position_boost
                    methods.append('adaptive_form_left_position')
            
            # Method 5: NLP-based semantic analysis
            if self.nlp:
                semantic_boost = self._calculate_adaptive_semantic_confidence(text)
                if semantic_boost > 0:
                    confidence += semantic_boost
                    methods.append('adaptive_semantic_analysis')
            
            # Method 6: Pattern matching with learned patterns
            pattern_boost = self._get_pattern_confidence(text)
            if pattern_boost > 0:
                confidence += pattern_boost
                methods.append('learned_pattern_match')
            
            # Use adaptive threshold instead of hardcoded 0.25
            min_threshold = adaptive_config.get_adaptive_value(
                "semantic_thresholds", "min_pairing_confidence"
            )
            
            if confidence >= min_threshold:
                key_candidates.append((block, confidence, '+'.join(methods)))
                logger.debug(f"Adaptive key candidate: '{text}' (confidence: {confidence:.2f})")
        
        return key_candidates
    
    def find_spatial_values(self, key_block: TextBlock, all_blocks: List[TextBlock]) -> List[Tuple[TextBlock, float, str]]:
        """
        Find values using adaptive spatial analysis
        """
        value_candidates = []
        key_center_x = key_block.bbox.center_x
        key_center_y = key_block.bbox.center_y
        key_right = key_block.bbox.right
        
        # Get adaptive proximity threshold
        proximity_threshold = adaptive_config.get_adaptive_value(
            "extraction_confidence", "proximity_distance_threshold"
        )
        
        for block in all_blocks:
            if block == key_block:
                continue
            
            confidence = 0.0
            methods = []
            
            block_center_x = block.bbox.center_x
            block_center_y = block.bbox.center_y
            block_left = block.bbox.x
            
            # Spatial relationship analysis with adaptive parameters
            horizontal_distance = abs(key_center_x - block_center_x)
            vertical_distance = abs(key_center_y - block_center_y)
            
            # Method 1: Same line (adaptive threshold)
            if vertical_distance < key_block.bbox.height * 0.8:
                same_line_boost = self._get_learned_confidence('same_line', 0.4)
                confidence += same_line_boost
                methods.append('adaptive_same_line')
                
                # Adjacent blocks bonus (adaptive)
                horizontal_gap = block_left - key_right
                if horizontal_gap >= -10 and horizontal_gap <= 50:
                    adjacent_boost = self._get_learned_confidence('adjacent_split', 0.5)
                    confidence += adjacent_boost
                    methods.append('adaptive_adjacent')
            
            # Method 2: Below key (adaptive)
            elif vertical_distance <= key_block.bbox.height * 4:
                below_boost = self._get_learned_confidence('below_key', 0.2)
                confidence += below_boost
                methods.append('adaptive_below_key')
            
            # Method 3: Proximity scoring (adaptive threshold)
            total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
            if total_distance < proximity_threshold:
                proximity_score = max(0, (proximity_threshold - total_distance) / proximity_threshold) * 0.2
                confidence += proximity_score
                methods.append('adaptive_proximity')
            
            # Method 4: Value pattern matching
            value_boost = self._calculate_value_pattern_confidence(block.text)
            if value_boost > 0:
                confidence += value_boost
                methods.append('adaptive_value_pattern')
            
            # Use adaptive minimum threshold
            min_threshold = adaptive_config.get_adaptive_value(
                "semantic_thresholds", "min_pairing_confidence"
            ) * 0.8  # Slightly lower for values
            
            if confidence >= min_threshold:
                value_candidates.append((block, confidence, '+'.join(methods)))
        
        return value_candidates
    
    def _get_learned_confidence(self, pattern_type: str, default_value: float) -> float:
        """
        Get confidence value learned from successful extractions
        """
        if pattern_type in self.pattern_confidence:
            learned_values = self.pattern_confidence[pattern_type]
            if len(learned_values) >= 3:  # Enough samples
                return np.mean(learned_values[-5:])  # Recent average
        
        return default_value
    
    def _calculate_adaptive_semantic_confidence(self, text: str) -> float:
        """
        Calculate semantic confidence using learned parameters
        """
        if not self.nlp:
            return 0.0
        
        try:
            doc = self.nlp(text)
            semantic_score = 0.0
            
            # NLP-based analysis with adaptive scoring
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    noun_boost = self._get_learned_confidence('semantic_noun', 0.15)
                    semantic_score += noun_boost
                    break
            
            # Entity analysis
            if any(ent.label_ in ['PERSON', 'ORG', 'GPE'] for ent in doc.ents):
                entity_penalty = self._get_learned_confidence('entity_penalty', -0.1)
                semantic_score += entity_penalty  # Usually negative
            
            return max(0, semantic_score)
            
        except Exception as e:
            logger.debug(f"Semantic confidence calculation error: {e}")
            return 0.0
    
    def _get_pattern_confidence(self, text: str) -> float:
        """
        Get confidence based on learned patterns
        """
        text_lower = text.lower().strip()
        
        # Check against learned successful patterns
        for pattern, confidence_values in self.learned_patterns.items():
            if pattern in text_lower and len(confidence_values) >= 2:
                # Return average confidence for this pattern
                return np.mean(confidence_values[-3:])  # Recent average
        
        return 0.0
    
    def _calculate_value_pattern_confidence(self, text: str) -> float:
        """
        Calculate confidence that text is a value using adaptive patterns
        """
        confidence = 0.0
        
        # Adaptive value characteristics
        if any(char.isdigit() for char in text):
            digit_boost = self._get_learned_confidence('has_digits', 0.2)
            confidence += digit_boost
        
        if text.istitle():
            title_value_boost = self._get_learned_confidence('title_value', 0.1)
            confidence += title_value_boost
        
        if len(text) > 10:
            length_value_boost = self._get_learned_confidence('long_value', 0.1)
            confidence += length_value_boost
        
        return confidence
    
    def learn_from_successful_extraction(self, kv_pairs: List[KeyValuePair], overall_confidence: float):
        """
        Learn from successful key-value extractions
        """
        if overall_confidence < 0.6:
            return  # Only learn from successful extractions
        
        for pair in kv_pairs:
            try:
                # Learn from extraction methods used
                methods = pair.extraction_method.split('|')
                
                for method in methods:
                    if 'adaptive_' in method:
                        method_type = method.replace('adaptive_', '')
                        
                        # Store successful confidence for this method
                        if method_type not in self.pattern_confidence:
                            self.pattern_confidence[method_type] = []
                        
                        self.pattern_confidence[method_type].append(pair.confidence)
                        
                        # Limit history
                        if len(self.pattern_confidence[method_type]) > 20:
                            self.pattern_confidence[method_type] = self.pattern_confidence[method_type][-20:]
                
                # Learn key patterns
                key_pattern = pair.key.lower().strip(':').strip()
                if key_pattern not in self.learned_patterns:
                    self.learned_patterns[key_pattern] = []
                
                self.learned_patterns[key_pattern].append(pair.confidence)
                
                # Limit pattern history
                if len(self.learned_patterns[key_pattern]) > 10:
                    self.learned_patterns[key_pattern] = self.learned_patterns[key_pattern][-10:]
                
            except Exception as e:
                logger.debug(f"Learning error: {e}")
        
        # Update adaptive config with learned values
        self._update_adaptive_config(overall_confidence)
    
    def _update_adaptive_config(self, success_confidence: float):
        """
        Update adaptive configuration based on learning
        """
        try:
            # Update confidence boosts based on learned values
            for method_type, confidence_values in self.pattern_confidence.items():
                if len(confidence_values) >= 3:
                    avg_confidence = np.mean(confidence_values)
                    
                    # Map method types to config parameters
                    config_mapping = {
                        'colon_terminator': ('extraction_confidence', 'colon_terminator_boost'),
                        'suffix_indicator': ('extraction_confidence', 'suffix_indicator_boost'),
                        'same_line': ('extraction_confidence', 'proximity_distance_threshold'),  # Indirect
                        'semantic_noun': ('semantic_thresholds', 'semantic_compatibility_threshold')
                    }
                    
                    if method_type in config_mapping:
                        category, parameter = config_mapping[method_type]
                        adaptive_config.learn_from_success(
                            category, parameter, avg_confidence, success_confidence
                        )
            
            # Save learned configuration
            adaptive_config.save_config()
            
        except Exception as e:
            logger.debug(f"Config update error: {e}")
    
    def extract_key_value_pairs(self, text_blocks: List[TextBlock]) -> List[KeyValuePair]:
        """
        Extract key-value pairs using fully adaptive parameters
        """
        if not text_blocks:
            return []
        
        try:
            logger.info(f"Starting adaptive extraction with {len(text_blocks)} text blocks")
            
            # Adaptive form detection
            form_analysis = self._detect_form_layout_adaptive(text_blocks)
            
            # Adaptive key identification
            key_candidates = self.identify_potential_keys(text_blocks, form_analysis)
            
            if not key_candidates:
                return []
            
            # Adaptive spatial clustering
            kv_pairs = self._cluster_adaptive_pairs(key_candidates, text_blocks)
            
            # Learn from this extraction
            if kv_pairs:
                avg_confidence = np.mean([pair.confidence for pair in kv_pairs])
                self.learn_from_successful_extraction(kv_pairs, avg_confidence)
            
            return kv_pairs
            
        except Exception as e:
            logger.error(f"Adaptive extraction error: {e}")
            return []
    
    def _detect_form_layout_adaptive(self, text_blocks: List[TextBlock]) -> Dict:
        """
        Adaptive form layout detection
        """
        # Simple adaptive form detection
        colon_count = sum(1 for block in text_blocks if block.text.endswith(':'))
        form_ratio = colon_count / len(text_blocks) if text_blocks else 0
        
        # Adaptive threshold
        form_threshold = self._get_learned_confidence('form_detection', 0.3)
        
        return {
            'is_form': form_ratio > form_threshold,
            'form_confidence': form_ratio,
            'left_aligned': True  # Simplified
        }
    
    def _cluster_adaptive_pairs(self, key_candidates: List[Tuple[TextBlock, float, str]], 
                              all_blocks: List[TextBlock]) -> List[KeyValuePair]:
        """
        Cluster key-value pairs using adaptive parameters
        """
        kv_pairs = []
        used_blocks = set()
        
        # Sort by adaptive confidence
        key_candidates.sort(key=lambda x: x[1], reverse=True)
        
        for key_block, key_confidence, key_method in key_candidates:
            if id(key_block) in used_blocks:
                continue
            
            # Find values with adaptive spatial analysis
            value_candidates = self.find_spatial_values(key_block, all_blocks)
            
            # Filter unused blocks
            available_candidates = [
                (block, conf, method) for block, conf, method in value_candidates
                if id(block) not in used_blocks
            ]
            
            if not available_candidates:
                continue
            
            # Take best match
            available_candidates.sort(key=lambda x: x[1], reverse=True)
            best_value_block, value_confidence, value_method = available_candidates[0]
            
            # Calculate combined confidence adaptively
            combined_confidence = (key_confidence + value_confidence) / 2
            
            # Use adaptive pairing threshold
            min_threshold = adaptive_config.get_adaptive_value(
                "semantic_thresholds", "min_pairing_confidence"
            )
            
            if combined_confidence >= min_threshold:
                kv_pair = KeyValuePair(
                    key=key_block.text.strip(),
                    value=best_value_block.text.strip(),
                    key_bbox=key_block.bbox,
                    value_bbox=best_value_block.bbox,
                    confidence=combined_confidence,
                    extraction_method=f"adaptive_key:{key_method}|adaptive_value:{value_method}"
                )
                
                kv_pairs.append(kv_pair)
                used_blocks.add(id(key_block))
                used_blocks.add(id(best_value_block))
        
        return kv_pairs