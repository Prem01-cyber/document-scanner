#!/usr/bin/env python3
"""
Visual Overlay Generator for Quality Assessment
Creates visual indicators for edge violations, text placement issues, and quality problems
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class QualityVisualizer:
    """Generate visual overlays for quality assessment results"""
    
    def __init__(self):
        self.colors = {
            'edge_cut': (255, 0, 0),        # Red for edge cuts
            'text_near_edge': (255, 165, 0), # Orange for text near edges
            'density_violation': (255, 255, 0), # Yellow for density violations
            'good_quality': (0, 255, 0),    # Green for good areas
            'blur_area': (128, 0, 128),     # Purple for blur
            'brightness_issue': (0, 255, 255) # Cyan for brightness
        }
        
    def create_quality_overlay(self, image: np.ndarray, quality_assessment: Dict, 
                              text_blocks: List = None) -> np.ndarray:
        """
        Create a visual overlay showing quality issues on the original image
        
        Args:
            image: Original image (numpy array)
            quality_assessment: Quality assessment results dictionary
            text_blocks: List of text blocks with bounding boxes
            
        Returns:
            Image with visual overlay highlighting quality issues
        """
        try:
            # Convert to PIL for easier drawing
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')
            
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Draw edge cut violations
            self._draw_edge_violations(draw, image.shape, quality_assessment, font)
            
            # Draw text-related issues
            if text_blocks:
                self._draw_text_issues(draw, text_blocks, quality_assessment, font, small_font)
            
            # Draw quality indicators
            self._draw_quality_indicators(draw, image.shape, quality_assessment, font, small_font)
            
            # Convert back to numpy array
            result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Error creating quality overlay: {e}")
            return image  # Return original image if overlay fails
    
    def _draw_edge_violations(self, draw: ImageDraw.Draw, image_shape: Tuple, 
                             quality_assessment: Dict, font):
        """Draw bounding boxes for edge cut violations"""
        try:
            height, width = image_shape[:2]
            cut_off_analysis = quality_assessment.get("cut_off_analysis", {})
            edge_cut_issues = cut_off_analysis.get("edge_cut_issues", [])
            
            # Define margin based on document type (from analysis)
            margin_pct = 0.03  # Default 3%
            margin_x = int(width * margin_pct)
            margin_y = int(height * margin_pct)
            
            # Draw edge violation boxes
            for issue in edge_cut_issues:
                color = self.colors['edge_cut']
                
                if issue == "left_edge_cut":
                    # Draw red box on left edge
                    draw.rectangle([0, 0, margin_x, height], outline=color, width=3)
                    draw.text((5, 10), "LEFT EDGE CUT", fill=color, font=font)
                    
                elif issue == "right_edge_cut":
                    # Draw red box on right edge
                    draw.rectangle([width - margin_x, 0, width, height], outline=color, width=3)
                    draw.text((width - 150, 10), "RIGHT EDGE CUT", fill=color, font=font)
                    
                elif issue == "top_edge_cut":
                    # Draw red box on top edge
                    draw.rectangle([0, 0, width, margin_y], outline=color, width=3)
                    draw.text((width//2 - 50, 5), "TOP EDGE CUT", fill=color, font=font)
                    
                elif issue == "bottom_edge_cut":
                    # Draw red box on bottom edge
                    draw.rectangle([0, height - margin_y, width, height], outline=color, width=3)
                    draw.text((width//2 - 60, height - 25), "BOTTOM EDGE CUT", fill=color, font=font)
            
            # Draw margin guidelines if there are edge issues
            if edge_cut_issues:
                guideline_color = (128, 128, 128)  # Gray
                # Draw dashed margin lines
                for x in range(0, width, 20):
                    draw.line([margin_x, x, margin_x, x+10], fill=guideline_color, width=1)
                    draw.line([width-margin_x, x, width-margin_x, x+10], fill=guideline_color, width=1)
                for y in range(0, height, 20):
                    draw.line([y, margin_y, y+10, margin_y], fill=guideline_color, width=1)
                    draw.line([y, height-margin_y, y+10, height-margin_y], fill=guideline_color, width=1)
                    
        except Exception as e:
            logger.debug(f"Error drawing edge violations: {e}")
    
    def _draw_text_issues(self, draw: ImageDraw.Draw, text_blocks: List, 
                         quality_assessment: Dict, font, small_font):
        """Draw indicators for text placement issues"""
        try:
            cut_off_analysis = quality_assessment.get("cut_off_analysis", {})
            text_edge_issues = cut_off_analysis.get("text_edge_issues", [])
            density_violations = cut_off_analysis.get("density_violations", [])
            
            # Draw bounding boxes around problematic text blocks
            for block in text_blocks:
                try:
                    # Extract coordinates
                    if hasattr(block, 'bbox'):
                        bbox = block.bbox
                        x = getattr(bbox, 'x', 0)
                        y = getattr(bbox, 'y', 0)
                        width = getattr(bbox, 'width', 0)
                        height = getattr(bbox, 'height', 0)
                    elif hasattr(block, 'x'):
                        x, y = block.x, block.y
                        width = getattr(block, 'width', 50)
                        height = getattr(block, 'height', 20)
                    else:
                        continue
                    
                    # Check if this text block is near edges
                    edge_threshold = 30
                    is_near_edge = (
                        x < edge_threshold or y < edge_threshold or 
                        x + width > draw.im.size[0] - edge_threshold or 
                        y + height > draw.im.size[1] - edge_threshold
                    )
                    
                    if is_near_edge:
                        # Draw orange box around text near edges
                        color = self.colors['text_near_edge']
                        draw.rectangle([x-2, y-2, x+width+2, y+height+2], outline=color, width=2)
                        
                        # Add small warning icon
                        draw.text((x, y-15), "⚠", fill=color, font=small_font)
                        
                except Exception as e:
                    logger.debug(f"Error processing text block: {e}")
                    continue
            
            # Draw density violation indicators
            if density_violations:
                # Add general density warning
                draw.text((10, 50), f"⚠ DENSITY VIOLATIONS: {len(density_violations)}", 
                         fill=self.colors['density_violation'], font=font)
                         
        except Exception as e:
            logger.debug(f"Error drawing text issues: {e}")
    
    def _draw_quality_indicators(self, draw: ImageDraw.Draw, image_shape: Tuple, 
                                quality_assessment: Dict, font, small_font):
        """Draw overall quality indicators and legend"""
        try:
            height, width = image_shape[:2]
            
            # Quality score indicator
            confidence = quality_assessment.get("confidence", 0)
            needs_rescan = quality_assessment.get("needs_rescan", False)
            
            # Choose color based on quality
            if confidence > 0.8:
                quality_color = self.colors['good_quality']
                quality_text = f"✅ EXCELLENT ({confidence:.2f})"
            elif confidence > 0.6:
                quality_color = (255, 255, 0)  # Yellow
                quality_text = f"⚠ GOOD ({confidence:.2f})"
            elif confidence > 0.4:
                quality_color = self.colors['text_near_edge']  # Orange
                quality_text = f"⚠ FAIR ({confidence:.2f})"
            else:
                quality_color = self.colors['edge_cut']  # Red
                quality_text = f"❌ POOR ({confidence:.2f})"
            
            # Draw quality badge in top-right corner
            badge_x = width - 200
            badge_y = 10
            draw.rectangle([badge_x-5, badge_y-5, badge_x+190, badge_y+25], 
                          fill=(0, 0, 0, 128), outline=quality_color, width=2)
            draw.text((badge_x, badge_y), quality_text, fill=quality_color, font=font)
            
            # Rescan recommendation
            if needs_rescan:
                rescan_decision = quality_assessment.get("rescan_decision", {})
                urgency = rescan_decision.get("rescan_urgency", "medium")
                urgency_colors = {"high": (255, 0, 0), "medium": (255, 165, 0), "low": (255, 255, 0)}
                urgency_color = urgency_colors.get(urgency, (255, 165, 0))
                
                draw.rectangle([badge_x-5, badge_y+30, badge_x+190, badge_y+55], 
                              fill=(0, 0, 0, 128), outline=urgency_color, width=2)
                draw.text((badge_x, badge_y+35), f"🔄 RESCAN: {urgency.upper()}", 
                         fill=urgency_color, font=font)
            
            # Advanced quality features indicator
            advanced_assessment = quality_assessment.get("advanced_quality_assessment", {})
            if advanced_assessment.get("advanced_features_available", False):
                quality_class = advanced_assessment.get("quality_class", "unknown")
                overall_score = advanced_assessment.get("overall_quality_score", 0)
                
                class_icons = {"excellent": "🌟", "good": "✅", "fair": "⚠️", "poor": "❌", "unknown": "❓"}
                class_icon = class_icons.get(quality_class, "❓")
                
                advanced_y = badge_y + (65 if needs_rescan else 35)
                draw.rectangle([badge_x-5, advanced_y-5, badge_x+190, advanced_y+20], 
                              fill=(0, 0, 0, 128), outline=(128, 128, 255), width=2)
                draw.text((badge_x, advanced_y), f"{class_icon} ADV: {quality_class.upper()}", 
                         fill=(128, 128, 255), font=font)
            
            # Legend for edge violations (if any exist)
            cut_off_analysis = quality_assessment.get("cut_off_analysis", {})
            edge_cut_issues = cut_off_analysis.get("edge_cut_issues", [])
            text_edge_issues = cut_off_analysis.get("text_edge_issues", [])
            
            if edge_cut_issues or text_edge_issues:
                legend_y = height - 80
                draw.rectangle([10, legend_y, 200, height-10], 
                              fill=(0, 0, 0, 180), outline=(255, 255, 255), width=1)
                draw.text((15, legend_y+5), "LEGEND:", fill=(255, 255, 255), font=font)
                
                if edge_cut_issues:
                    draw.rectangle([15, legend_y+25, 25, legend_y+35], fill=self.colors['edge_cut'])
                    draw.text((30, legend_y+25), "Edge Cut", fill=(255, 255, 255), font=small_font)
                
                if text_edge_issues:
                    draw.rectangle([15, legend_y+40, 25, legend_y+50], fill=self.colors['text_near_edge'])
                    draw.text((30, legend_y+40), "Text Near Edge", fill=(255, 255, 255), font=small_font)
                    
        except Exception as e:
            logger.debug(f"Error drawing quality indicators: {e}")

# Global instance for easy access
quality_visualizer = QualityVisualizer()


def create_quality_overlay(image: np.ndarray, quality_assessment: Dict, 
                          text_blocks: List = None) -> np.ndarray:
    """
    Convenience function to create quality overlay
    """
    return quality_visualizer.create_quality_overlay(image, quality_assessment, text_blocks)