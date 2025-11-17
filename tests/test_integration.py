"""Integration tests for the complete workflow."""

import json
import tempfile
from pathlib import Path

import pytest

from baseline_generator import RuleParser, RuleSchema, BaselineEngine


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_text_to_json_to_baseline(self):
        """Test complete workflow from text rules to baseline generation."""
        # Step 1: Parse text rules
        parser = RuleParser()
        rules_text = """
If building type is office and building area is less than 25000 sqft then set lighting power density to 1.0 W/sqft
If climate zone is 5a then set heating efficiency to 0.8
"""
        rules = parser.parse_rules_from_text(rules_text, category="test")
        
        assert len(rules) == 2
        
        # Step 2: Create rule schema
        schema = RuleSchema(version="1.0", rules=rules)
        schema_dict = schema.to_dict()
        
        assert schema_dict["version"] == "1.0"
        assert len(schema_dict["rules"]) == 2
        
        # Step 3: Load schema from dict (simulating JSON save/load)
        loaded_schema = RuleSchema.from_dict(schema_dict)
        
        assert len(loaded_schema.rules) == 2
        
        # Step 4: Generate baseline
        engine = BaselineEngine(loaded_schema)
        building_spec = {
            "building_type": "office",
            "building_area": 15000,
            "climate_zone": "5a"
        }
        
        baseline = engine.generate_baseline(building_spec)
        
        assert "matched_rules" in baseline
        assert "baseline_properties" in baseline
        assert len(baseline["matched_rules"]) == 2
    
    def test_json_rules_file_workflow(self, tmp_path):
        """Test workflow using JSON rules file."""
        # Create rules JSON file
        rules_data = {
            "version": "1.0",
            "rules": [
                {
                    "id": "test_001",
                    "name": "Test Rule",
                    "description": "A test rule",
                    "category": "test",
                    "conditions": {
                        "field": "building_type",
                        "operator": "equals",
                        "value": "office"
                    },
                    "actions": [
                        {
                            "action_type": "set_value",
                            "target": "test_property",
                            "value": 100
                        }
                    ]
                }
            ]
        }
        
        rules_file = tmp_path / "rules.json"
        with open(rules_file, 'w') as f:
            json.dump(rules_data, f)
        
        # Load rules
        with open(rules_file) as f:
            loaded_data = json.load(f)
        
        schema = RuleSchema.from_dict(loaded_data)
        
        # Generate baseline
        engine = BaselineEngine(schema)
        building_spec = {"building_type": "office"}
        baseline = engine.generate_baseline(building_spec)
        
        assert baseline["baseline_properties"]["test_property"] == 100
    
    def test_multiple_categories(self):
        """Test rules from different categories."""
        from baseline_generator.schema import (
            Rule, Condition, Action, ComparisonOperator
        )
        
        # Create rules for different categories
        lighting_rule = Rule(
            id="l001",
            name="Lighting Rule",
            description="Test lighting rule",
            category="lighting",
            conditions=Condition("building_type", ComparisonOperator.EQUALS, "office"),
            actions=[Action("set_value", "lighting_power_density", 1.0)]
        )
        
        hvac_rule = Rule(
            id="h001",
            name="HVAC Rule",
            description="Test HVAC rule",
            category="hvac",
            conditions=Condition("building_type", ComparisonOperator.EQUALS, "office"),
            actions=[Action("set_value", "cooling_cop", 3.5)]
        )
        
        schema = RuleSchema(version="1.0", rules=[lighting_rule, hvac_rule])
        engine = BaselineEngine(schema)
        
        building_spec = {"building_type": "office"}
        baseline = engine.generate_baseline(building_spec)
        
        # Both rules should match
        assert len(baseline["matched_rules"]) == 2
        assert baseline["baseline_properties"]["lighting_power_density"] == 1.0
        assert baseline["baseline_properties"]["cooling_cop"] == 3.5
    
    def test_parser_with_various_formats(self):
        """Test parser with different rule text formats."""
        parser = RuleParser()
        
        test_cases = [
            "If building area is greater than 10000 then set value to 1.5",
            "If climate zone is 5a then apply method test-method",
            "If building type is office and climate zone is 5a then set lighting to 1.0",
        ]
        
        for i, text in enumerate(test_cases):
            rule = parser.parse_rule_text(text, rule_id=f"test_{i:03d}")
            assert rule.id == f"test_{i:03d}"
            assert len(rule.actions) > 0
    
    def test_validation_workflow(self):
        """Test building spec validation."""
        from baseline_generator.schema import (
            Rule, Condition, Action, ComparisonOperator
        )
        
        rule = Rule(
            id="v001",
            name="Validation Rule",
            description="Test validation",
            category="test",
            conditions=Condition("required_field", ComparisonOperator.EQUALS, "value"),
            actions=[Action("set_value", "result", "ok")]
        )
        
        schema = RuleSchema(version="1.0", rules=[rule])
        engine = BaselineEngine(schema)
        
        # Valid spec
        valid_spec = {"required_field": "value"}
        result = engine.validate_building_spec(valid_spec)
        assert result["valid"] is True
        
        # Missing field
        invalid_spec = {}
        result = engine.validate_building_spec(invalid_spec)
        assert len(result["warnings"]) > 0
