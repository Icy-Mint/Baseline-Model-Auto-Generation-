"""
Text-to-JSON parser for building energy rules.

This module provides functionality to parse natural language building energy rules
into structured JSON format. Uses pattern matching and NLP techniques without
relying on copyrighted content.
"""

import re
from typing import Dict, Any, List, Optional, Union
from .schema import (
    Rule, Condition, ConditionGroup, Action,
    ComparisonOperator, LogicalOperator
)


class RuleParser:
    """Parser for converting text-based rules to structured JSON."""
    
    # Pattern mappings for common rule structures
    OPERATOR_PATTERNS = {
        r'\bequals?\b|\bis\b|\b=\b': ComparisonOperator.EQUALS,
        r'\bnot equal(s)?\b|\bis not\b|\b!=\b': ComparisonOperator.NOT_EQUALS,
        r'\bgreater than or equal\b|\b>=\b|\bat least\b': ComparisonOperator.GREATER_THAN_OR_EQUAL,
        r'\bless than or equal\b|\b<=\b|\bat most\b': ComparisonOperator.LESS_THAN_OR_EQUAL,
        r'\bgreater than\b|\b>\b|\bmore than\b|\bexceeds\b': ComparisonOperator.GREATER_THAN,
        r'\bless than\b|\b<\b|\bbelow\b': ComparisonOperator.LESS_THAN,
    }
    
    LOGICAL_OPERATORS = {
        r'\band\b': LogicalOperator.AND,
        r'\bor\b': LogicalOperator.OR,
    }
    
    # Common building property patterns
    PROPERTY_PATTERNS = {
        r'building\s+area': 'building_area',
        r'floor\s+area': 'floor_area',
        r'climate\s+zone': 'climate_zone',
        r'building\s+type': 'building_type',
        r'number\s+of\s+stories': 'num_stories',
        r'lighting\s+power\s+density': 'lighting_power_density',
        r'window\s+to\s+wall\s+ratio': 'window_to_wall_ratio',
        r'u-?factor': 'u_factor',
        r'r-?value': 'r_value',
        r'shgc': 'shgc',
        r'cop': 'cop',
        r'efficiency': 'efficiency',
    }
    
    # Unit patterns
    UNIT_PATTERNS = {
        r'sq\.?\s*ft\.?|square\s+feet?': 'sqft',
        r'sq\.?\s*m\.?|square\s+meters?': 'sqm',
        r'w/sq\.?\s*ft\.?|watts?\s+per\s+square\s+foot': 'W/sqft',
        r'w/sq\.?\s*m\.?|watts?\s+per\s+square\s+meter': 'W/sqm',
        r'°?f\b|fahrenheit': 'degF',
        r'°?c\b|celsius': 'degC',
        r'btu': 'BTU',
        r'percent|%': 'percent',
    }
    
    def __init__(self):
        """Initialize the parser."""
        self.rule_counter = 0
    
    def parse_rule_text(self, text: str, rule_id: Optional[str] = None,
                       category: str = "general") -> Rule:
        """Parse a text-based rule into a Rule object.
        
        Args:
            text: Natural language rule text
            rule_id: Optional rule ID (auto-generated if not provided)
            category: Category for the rule
            
        Returns:
            Parsed Rule object
        """
        if rule_id is None:
            self.rule_counter += 1
            rule_id = f"rule_{self.rule_counter:03d}"
        
        # Extract conditions and actions from text
        conditions = self._extract_conditions(text)
        actions = self._extract_actions(text)
        
        # Generate name and description
        name = self._generate_rule_name(text)
        description = text.strip()
        
        return Rule(
            id=rule_id,
            name=name,
            description=description,
            category=category,
            conditions=conditions,
            actions=actions,
            priority=0
        )
    
    def _extract_conditions(self, text: str) -> Union[Condition, ConditionGroup]:
        """Extract conditions from rule text."""
        text_lower = text.lower()
        
        # Check for logical operators to determine if we need a ConditionGroup
        has_and = re.search(r'\band\b', text_lower)
        has_or = re.search(r'\bor\b', text_lower)
        
        # Split by "then", "set", "apply", etc. to separate conditions from actions
        condition_text = text
        for separator in [' then ', ' set ', ' apply ', ' use ']:
            if separator in text_lower:
                condition_text = text_lower.split(separator)[0]
                break
        
        if has_and or has_or:
            return self._parse_condition_group(condition_text)
        else:
            return self._parse_single_condition(condition_text)
    
    def _parse_single_condition(self, text: str) -> Condition:
        """Parse a single condition from text."""
        text_lower = text.lower()
        
        # Find the property
        field = "property"
        for pattern, prop_name in self.PROPERTY_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                field = prop_name
                break
        
        # Find the operator (check in specific order - longer patterns first)
        operator = ComparisonOperator.EQUALS
        operator_patterns_ordered = [
            (r'\bgreater than or equal\b|\b>=\b|\bat least\b', ComparisonOperator.GREATER_THAN_OR_EQUAL),
            (r'\bless than or equal\b|\b<=\b|\bat most\b', ComparisonOperator.LESS_THAN_OR_EQUAL),
            (r'\bnot equal(s)?\b|\bis not\b|\b!=\b', ComparisonOperator.NOT_EQUALS),
            (r'\bgreater than\b|\b>\b|\bmore than\b|\bexceeds\b', ComparisonOperator.GREATER_THAN),
            (r'\bless than\b|\b<\b|\bbelow\b', ComparisonOperator.LESS_THAN),
            (r'\bequals?\b|\bis\b|\b=\b', ComparisonOperator.EQUALS),
        ]
        for pattern, op in operator_patterns_ordered:
            if re.search(pattern, text_lower, re.IGNORECASE):
                operator = op
                break
        
        # Extract value (look for numbers or quoted strings)
        value: Union[str, int, float] = ""
        
        # Try to find numeric values
        number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', text_lower)
        if number_match:
            value_str = number_match.group(1)
            value = float(value_str) if '.' in value_str else int(value_str)
        else:
            # Try to find quoted or specific values
            quoted_match = re.search(r'["\']([^"\']+)["\']', text)
            if quoted_match:
                value = quoted_match.group(1)
            else:
                # Look for common building types or climate zones
                if 'office' in text_lower:
                    value = 'office'
                elif 'retail' in text_lower:
                    value = 'retail'
                elif 'residential' in text_lower:
                    value = 'residential'
                # Check for climate zone patterns (more flexible)
                climate_match = re.search(r'(?:zone\s+|is\s+)(\d+[a-z]?)', text_lower)
                if climate_match:
                    value = climate_match.group(1)
        
        # Extract unit (check after value to avoid consuming numbers)
        unit = None
        for pattern, unit_name in self.UNIT_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                unit = unit_name
                break
        
        return Condition(
            field=field,
            operator=operator,
            value=value,
            unit=unit
        )
    
    def _parse_condition_group(self, text: str) -> ConditionGroup:
        """Parse a group of conditions connected by logical operators."""
        text_lower = text.lower()
        
        # Determine the primary logical operator
        operator = LogicalOperator.AND
        if ' or ' in text_lower:
            operator = LogicalOperator.OR
        
        # Split by the operator
        separator = ' and ' if operator == LogicalOperator.AND else ' or '
        parts = text_lower.split(separator)
        
        conditions = []
        for part in parts:
            conditions.append(self._parse_single_condition(part))
        
        return ConditionGroup(
            operator=operator,
            conditions=conditions
        )
    
    def _extract_actions(self, text: str) -> List[Action]:
        """Extract actions from rule text."""
        text_lower = text.lower()
        actions = []
        
        # Look for action keywords
        if ' then ' in text_lower or ' set ' in text_lower:
            action_text = text_lower
            for separator in [' then ', ' set ']:
                if separator in text_lower:
                    action_text = text_lower.split(separator, 1)[1]
                    break
            
            # Determine action type and extract parameters
            action_type = "set_value"
            target = "property"
            value: Union[str, int, float] = ""
            
            # Find target property
            for pattern, prop_name in self.PROPERTY_PATTERNS.items():
                if re.search(pattern, action_text, re.IGNORECASE):
                    target = prop_name
                    break
            
            # Extract value
            number_match = re.search(r'\b(\d+(?:\.\d+)?)\b', action_text)
            if number_match:
                value_str = number_match.group(1)
                value = float(value_str) if '.' in value_str else int(value_str)
            else:
                # Look for method names or references
                if 'method' in action_text or 'procedure' in action_text:
                    action_type = "apply_method"
                    # Extract method name
                    method_match = re.search(r'(method|procedure)\s+([a-z0-9_-]+)', action_text)
                    if method_match:
                        value = method_match.group(2)
                elif 'table' in action_text:
                    action_type = "reference_table"
                    table_match = re.search(r'table\s+([a-z0-9_.-]+)', action_text)
                    if table_match:
                        value = table_match.group(1)
            
            actions.append(Action(
                action_type=action_type,
                target=target,
                value=value
            ))
        
        # If no actions found, create a default action
        if not actions:
            actions.append(Action(
                action_type="evaluate",
                target="compliance",
                value="check"
            ))
        
        return actions
    
    def _generate_rule_name(self, text: str) -> str:
        """Generate a concise name for the rule."""
        # Take first few words or extract key concept
        words = text.split()[:6]
        name = ' '.join(words)
        if len(text.split()) > 6:
            name += '...'
        return name
    
    def parse_rules_from_text(self, text: str, category: str = "general") -> List[Rule]:
        """Parse multiple rules from text (one rule per line or paragraph).
        
        Args:
            text: Text containing multiple rules
            category: Category for all rules
            
        Returns:
            List of parsed Rule objects
        """
        # Split by double newlines (paragraphs) or single newlines
        lines = text.strip().split('\n\n')
        if len(lines) == 1:
            lines = text.strip().split('\n')
        
        rules = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    rule = self.parse_rule_text(line, category=category)
                    rules.append(rule)
                except Exception:
                    # Skip lines that can't be parsed
                    continue
        
        return rules
