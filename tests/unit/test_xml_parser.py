"""
Tests for XMLOutputParser.

Following production-flow protocol:
- Use REAL test data (actual LLM-style outputs)
- Cover normal cases, edge cases, and error cases
- Test all input/output scenarios
- Comprehensive coverage
"""

import pytest
import logging
from chaos_auto_prompt.utils.xml_parser import (
    XMLOutputParser,
    extract_construction,
    extract_reasoning,
)


class TestXMLOutputParser:
    """Test suite for XMLOutputParser with real-world scenarios."""

    def test_normal_case_simple_construction_tag(self):
        """Test Case 1: Well-formed single construction tag."""
        # Real LLM-style output
        text = """
        Based on the problem, I'll analyze it step by step.

        <construction>
        Entities: user, task, deadline, priority
        State Variables:
        - task_status: enum [pending, in_progress, completed]
        - user_availability: boolean
        Actions:
        - assign_task: requires user available AND task pending
        - complete_task: requires task in_progress
        Constraints:
        - One user can only work on one task at a time
        - Tasks must be completed before deadline
        </construction>

        Now let me use this model to solve the problem.
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is True
        assert "Entities: user, task, deadline, priority" in result["content"]
        assert "task_status: enum" in result["content"]
        assert result["metadata"]["tag_found"] is True
        assert result["metadata"]["num_occurrences"] == 1
        assert result["metadata"]["extraction_method"] in ["regex", "xml_parser"]

    def test_normal_case_think_tag(self):
        """Test Case 2: Well-formed reasoning/think tag."""
        text = """
        <think>
        Step 1: Analyze the problem constraints
        - We have 3 tasks and 2 users
        - Each task has different priority

        Step 2: Apply the construction model
        - Check user_availability for both users
        - User1 is available, User2 is busy

        Step 3: Make decision
        - Assign highest priority task to User1
        - Queue remaining tasks

        Final answer: Assign Task A to User1
        </think>
        """

        result = XMLOutputParser.extract(text, "think")

        assert result["success"] is True
        assert "Step 1: Analyze the problem constraints" in result["content"]
        assert "Final answer: Assign Task A to User1" in result["content"]
        assert result["metadata"]["num_occurrences"] == 1

    def test_edge_case_multiple_occurrences(self):
        """Test Case 3: Multiple occurrences - should take first and log warning."""
        text = """
        <construction>First version: entities A, B, C</construction>
        Some text in between.
        <construction>Second version: entities X, Y, Z</construction>
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is True
        assert "First version" in result["content"]
        assert "Second version" not in result["content"]
        assert result["metadata"]["num_occurrences"] == 2

    def test_edge_case_nested_tags(self):
        """Test Case 4: Nested tags - should extract full content."""
        text = """
        <construction>
        Outer content
        <inner>Nested content here</inner>
        More outer content
        </construction>
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is True
        assert "Outer content" in result["content"]
        assert "Nested content here" in result["content"]
        assert "More outer content" in result["content"]

    def test_edge_case_tag_with_attributes(self):
        """Test Case 5: Tags with attributes should still parse."""
        text = """
        <construction type="v1" model="gpt-4">
        Entities: user, task
        State: active
        </construction>
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is True
        assert "Entities: user, task" in result["content"]
        assert "State: active" in result["content"]

    def test_edge_case_empty_tag(self):
        """Test Case 6: Empty tag should return success with empty content."""
        text = """
        Some text before
        <construction></construction>
        Some text after
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is True
        assert result["content"] == ""
        assert result["metadata"]["tag_found"] is True

    def test_edge_case_multiline_content(self):
        """Test Case 7: Multiline content with special characters."""
        text = """
        <think>
        This is a complex reasoning path with:
        - Bullet points
        - Multiple paragraphs

        Special characters: & < > " '
        Math: 2 + 2 = 4
        Code: if (x > 5) { return true; }

        Conclusion: The answer is 42.
        </think>
        """

        result = XMLOutputParser.extract(text, "think")

        assert result["success"] is True
        assert "Bullet points" in result["content"]
        assert "Special characters" in result["content"]
        assert "Conclusion: The answer is 42." in result["content"]

    def test_edge_case_whitespace_preservation(self):
        """Test Case 8: Whitespace preservation option."""
        text = "<construction>   Content with spaces   </construction>"

        # Test with preserve_whitespace=False (default)
        result_stripped = XMLOutputParser.extract(text, "construction", preserve_whitespace=False)
        assert result_stripped["content"] == "Content with spaces"

        # Test with preserve_whitespace=True
        result_preserved = XMLOutputParser.extract(text, "construction", preserve_whitespace=True)
        assert result_preserved["content"] == "   Content with spaces   "

    def test_edge_case_xml_entities(self):
        """Test Case 9: XML entities should be unescaped."""
        text = """
        <construction>
        Entities: user &amp; task
        Condition: value &lt; 10
        Quote: &quot;Hello World&quot;
        </construction>
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is True
        assert "user & task" in result["content"]
        assert "value < 10" in result["content"]
        assert '"Hello World"' in result["content"]

    def test_error_case_tag_not_found(self):
        """Test Case 10: Tag not found should return error."""
        text = """
        This is some text without any XML tags.
        Just plain text here.
        Nothing to extract.
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is False
        assert result["content"] == ""
        assert "error" in result
        assert "not found" in result["error"].lower()
        assert result["metadata"]["tag_found"] is False
        assert result["metadata"]["num_occurrences"] == 0

    def test_error_case_empty_text(self):
        """Test Case 11: Empty text should raise ValueError."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            XMLOutputParser.extract("", "construction")

    def test_error_case_empty_tag_name(self):
        """Test Case 12: Empty tag name should raise ValueError."""
        with pytest.raises(ValueError, match="Tag name cannot be empty"):
            XMLOutputParser.extract("Some text", "")

    def test_error_case_invalid_tag_name(self):
        """Test Case 13: Invalid tag name should raise ValueError."""
        invalid_names = [
            "tag with spaces",
            "tag@symbol",
            "tag#hash",
            "tag.dot",
            "tag/slash"
        ]

        for invalid_name in invalid_names:
            with pytest.raises(ValueError, match="Invalid tag name"):
                XMLOutputParser.extract("Some text", invalid_name)

    def test_edge_case_case_insensitive(self):
        """Test Case 14: Tag matching should be case-insensitive."""
        text = """
        <CONSTRUCTION>Upper case tag</CONSTRUCTION>
        <Construction>Mixed case tag</Construction>
        """

        result = XMLOutputParser.extract(text, "construction")

        assert result["success"] is True
        assert "Upper case tag" in result["content"] or "Mixed case tag" in result["content"]

    def test_edge_case_unclosed_tag_graceful_handling(self):
        """Test Case 15: Unclosed tag should fail gracefully."""
        text = """
        <construction>
        This tag is not properly closed
        Missing the closing tag
        """

        result = XMLOutputParser.extract(text, "construction")

        # Should fail gracefully (not crash)
        assert result["success"] is False
        assert "error" in result or result["content"] == ""

    def test_edge_case_malformed_xml(self):
        """Test Case 16: Malformed XML should be handled by regex."""
        text = """
        <construction>
        This is < malformed > XML with < random brackets
        But it should still extract
        </construction>
        """

        result = XMLOutputParser.extract(text, "construction")

        # Regex should handle this
        assert result["success"] is True
        assert "malformed" in result["content"].lower()

    def test_integration_real_llm_output_construction(self):
        """Integration Test 1: Real Gemini-style construction output."""
        text = """
        I'll analyze this scheduling problem step by step.

        <construction>
        (1) Relevant Entities:
        - Person: Alice, Bob, Carol (meeting participants)
        - TimeSlot: 9AM-10AM, 10AM-11AM, 2PM-3PM (available slots)
        - Meeting: the scheduled event
        - Calendar: tracks availability

        (2) State Variables:
        - person_availability: dict[Person, list[TimeSlot]]
          Initial: {Alice: [9AM, 10AM], Bob: [10AM, 2PM], Carol: [9AM, 2PM]}
        - meeting_scheduled: boolean (initially False)
        - selected_time: TimeSlot | None

        (3) Possible Actions:
        Action: schedule_meeting(time_slot)
        Preconditions:
          - meeting_scheduled == False
          - time_slot in ALL participants' availability lists
        Effects:
          - meeting_scheduled = True
          - selected_time = time_slot
          - Remove time_slot from all calendars

        (4) Constraints:
        - All 3 people must be available at the selected time
        - Can only schedule one meeting
        - Time slot must be within business hours (9AM-5PM)
        - No double-booking allowed
        </construction>

        Now I'll use this model to find a solution.
        """

        result = extract_construction(text)

        assert result["success"] is True
        assert "(1) Relevant Entities" in result["content"]
        assert "Person: Alice, Bob, Carol" in result["content"]
        assert "(2) State Variables" in result["content"]
        assert "person_availability" in result["content"]
        assert "(3) Possible Actions" in result["content"]
        assert "(4) Constraints" in result["content"]
        assert result["metadata"]["num_occurrences"] == 1

    def test_integration_real_llm_output_reasoning(self):
        """Integration Test 2: Real Gemini-style reasoning output."""
        text = """
        <think>
        Let me work through this systematically using the constructed model.

        Step 1: Check availability constraints
        From the state variables:
        - Alice is available: [9AM-10AM, 10AM-11AM]
        - Bob is available: [10AM-11AM, 2PM-3PM]
        - Carol is available: [9AM-10AM, 2PM-3PM]

        Step 2: Find common time slots
        Looking for overlap in all three availability lists:
        - 9AM-10AM: Alice ✓, Bob ✗, Carol ✓ → NOT valid
        - 10AM-11AM: Alice ✓, Bob ✓, Carol ✗ → NOT valid
        - 2PM-3PM: Alice ✗, Bob ✓, Carol ✓ → NOT valid

        Step 3: Verify constraints
        Checking constraint: "All 3 people must be available"
        - No single time slot satisfies this constraint

        Step 4: Conclusion
        Given the current availability, it's IMPOSSIBLE to schedule a meeting
        that satisfies all constraints. No valid time slot exists.

        Therefore, the answer is: "Cannot schedule - no common available time"
        </think>

        My final answer is that this meeting cannot be scheduled with current constraints.
        """

        result = extract_reasoning(text)

        assert result["success"] is True
        assert "Step 1: Check availability constraints" in result["content"]
        assert "Step 2: Find common time slots" in result["content"]
        assert "Step 3: Verify constraints" in result["content"]
        assert "Step 4: Conclusion" in result["content"]
        assert "Cannot schedule" in result["content"]

    def test_performance_large_text(self):
        """Performance Test: Handle large text efficiently."""
        # Generate large text (10,000 characters)
        large_content = "Line of content.\n" * 500
        text = f"<construction>{large_content}</construction>"

        import time
        start = time.time()
        result = XMLOutputParser.extract(text, "construction")
        elapsed = time.time() - start

        assert result["success"] is True
        assert len(result["content"]) > 5000
        assert elapsed < 0.1  # Should be < 100ms

    def test_convenience_functions(self):
        """Test Case 17: Convenience functions work correctly."""
        text_construction = "<construction>Test construction</construction>"
        text_reasoning = "<think>Test reasoning</think>"

        result_construction = extract_construction(text_construction)
        result_reasoning = extract_reasoning(text_reasoning)

        assert result_construction["success"] is True
        assert result_construction["content"] == "Test construction"

        assert result_reasoning["success"] is True
        assert result_reasoning["content"] == "Test reasoning"


class TestXMLOutputParserLogging:
    """Test logging behavior."""

    def test_logging_on_success(self, caplog):
        """Verify appropriate logging on successful extraction."""
        with caplog.at_level(logging.INFO):
            text = "<construction>Test content</construction>"
            XMLOutputParser.extract(text, "construction")

            assert "Successfully extracted tag" in caplog.text

    def test_logging_on_failure(self, caplog):
        """Verify appropriate logging on failed extraction."""
        with caplog.at_level(logging.WARNING):
            text = "No tags here"
            XMLOutputParser.extract(text, "construction")

            assert "not found" in caplog.text.lower()

    def test_logging_multiple_occurrences(self, caplog):
        """Verify warning logged for multiple occurrences."""
        with caplog.at_level(logging.WARNING):
            text = "<construction>First</construction><construction>Second</construction>"
            XMLOutputParser.extract(text, "construction")

            assert "occurrences" in caplog.text.lower()
            assert "first occurrence" in caplog.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
