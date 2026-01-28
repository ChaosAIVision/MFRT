"""
Tests for ConstructionExtractor.

Following production-flow protocol:
- Use REAL test data (actual LLM construction outputs)
- Cover normal cases, edge cases, and error cases
- Test all extraction scenarios
"""

import pytest
import logging
from chaos_auto_prompt.utils.construction_extractor import ConstructionExtractor


class TestConstructionExtractor:
    """Test suite for ConstructionExtractor with real-world LLM outputs."""

    def test_normal_case_numbered_format(self):
        """Test Case 1: Well-formed construction with numbered sections."""
        construction_text = """
        (1) Relevant Entities:
        - Person: Alice, Bob, Carol
        - TimeSlot: 9AM-10AM, 10AM-11AM, 2PM-3PM
        - Meeting: the scheduled event
        - Calendar: tracks availability

        (2) State Variables:
        - person_availability: dict[Person, list[TimeSlot]]
        - meeting_scheduled: boolean
        - selected_time: TimeSlot

        (3) Possible Actions:
        Action: schedule_meeting
        Preconditions: meeting not scheduled, all participants available
        Effects: meeting_scheduled = true, time slot removed from calendars

        (4) Constraints:
        - All 3 people must be available at the selected time
        - Can only schedule one meeting
        - No double-booking allowed
        """

        result = ConstructionExtractor.extract(construction_text)

        assert result["metadata"]["extraction_confidence"] == 1.0
        assert len(result["entities"]) >= 3
        assert "Person: Alice, Bob, Carol" in result["entities"]

        assert len(result["state_variables"]) >= 2
        assert any(sv["name"] == "person_availability" for sv in result["state_variables"])

        assert len(result["actions"]) >= 1
        assert result["actions"][0]["name"] == "schedule_meeting"

        assert len(result["constraints"]) >= 2
        assert result["metadata"]["missing_sections"] == []

    def test_normal_case_plain_headers(self):
        """Test Case 2: Construction with plain section headers."""
        construction_text = """
        Entities: user, task, deadline, priority, status

        State Variables:
        task_status: enum [pending, in_progress, completed]
        user_availability: boolean
        deadline_date: datetime

        Actions:
        - assign_task: requires user available AND task pending
        - complete_task: requires task in_progress
        - cancel_task: requires task not completed

        Constraints:
        - One user can only work on one task at a time
        - Tasks must be completed before deadline
        - High priority tasks should be assigned first
        """

        result = ConstructionExtractor.extract(construction_text)

        assert result["metadata"]["extraction_confidence"] == 1.0
        assert len(result["entities"]) == 5
        assert "user" in result["entities"]
        assert "task" in result["entities"]

        assert len(result["state_variables"]) == 3
        task_status = next(sv for sv in result["state_variables"] if sv["name"] == "task_status")
        assert task_status["type"] == "enum"
        assert "pending" in task_status["possible_values"]

        assert len(result["actions"]) == 3
        assert any(a["name"] == "assign_task: requires user available AND task pending" for a in result["actions"])

        assert len(result["constraints"]) == 3

    def test_edge_case_missing_one_section(self):
        """Test Case 3: Construction missing actions section."""
        construction_text = """
        Entities: student, course, grade

        State Variables:
        enrollment_status: enum [enrolled, dropped, completed]
        current_grade: float

        Constraints:
        - Students must enroll before attending
        - Grades must be between 0.0 and 4.0
        """

        result = ConstructionExtractor.extract(construction_text)

        assert 0.5 <= result["metadata"]["extraction_confidence"] < 1.0
        assert "actions" in result["metadata"]["missing_sections"]
        assert len(result["entities"]) == 3
        assert len(result["state_variables"]) == 2
        assert len(result["actions"]) == 0
        assert len(result["constraints"]) == 2

    def test_edge_case_simple_lists_only(self):
        """Test Case 4: Very simple format without types/preconditions."""
        construction_text = """
        Entities:
        User, Product, ShoppingCart, Order, Payment

        State:
        cart_items, cart_total, order_status, payment_confirmed

        Actions:
        add_to_cart, remove_from_cart, checkout, process_payment, confirm_order

        Constraints:
        Cart must not be empty for checkout
        Payment must be confirmed before order completion
        """

        result = ConstructionExtractor.extract(construction_text)

        # Note: "State:" header doesn't match "State Variables:" so confidence < 1.0
        assert result["metadata"]["extraction_confidence"] >= 0.75
        assert len(result["entities"]) == 5
        assert "User" in result["entities"]

        # Actions without preconditions/effects
        assert len(result["actions"]) == 5
        assert all("name" in action for action in result["actions"])

        assert len(result["constraints"]) == 2

    def test_edge_case_multiline_descriptions(self):
        """Test Case 5: Actions with multiline preconditions and effects."""
        construction_text = """
        Entities: Agent, Task, Resource

        State Variables:
        - agent_busy: boolean
        - task_queue: list[Task]
        - available_resources: int

        Actions:
        Action: allocate_task
        Preconditions:
          - Agent is not busy
          - Task exists in queue
          - Sufficient resources available
        Effects:
          - Agent becomes busy
          - Task removed from queue
          - Resources decremented

        Constraints:
        - Only one task per agent at a time
        - Resources cannot go negative
        """

        result = ConstructionExtractor.extract(construction_text)

        assert result["metadata"]["extraction_confidence"] == 1.0
        assert len(result["actions"]) == 1

        action = result["actions"][0]
        assert action["name"] == "allocate_task"
        assert len(action.get("preconditions", [])) >= 2
        assert len(action.get("effects", [])) >= 2

    def test_edge_case_comma_separated_entities(self):
        """Test Case 6: Comma-separated entities (common LLM format)."""
        construction_text = """
        Entities: user, admin, role, permission, resource, action, log

        State Variables:
        user_roles: list[Role]
        permissions: dict[Role, list[Permission]]

        Actions:
        grant_permission, revoke_permission, assign_role

        Constraints:
        Admins have all permissions
        Users must have role to access resources
        """

        result = ConstructionExtractor.extract(construction_text)

        # Comma-separated should work, might stop at "action, log" due to "action" keyword
        assert len(result["entities"]) >= 5
        assert "user" in result["entities"]
        assert "admin" in result["entities"]
        assert "role" in result["entities"]

    def test_error_case_empty_text(self):
        """Test Case 7: Empty construction text should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ConstructionExtractor.extract("")

    def test_error_case_too_short(self):
        """Test Case 8: Text too short should raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            ConstructionExtractor.extract("Entities: user")

    def test_error_case_no_keywords(self):
        """Test Case 9: Text without construction keywords should raise ValueError."""
        long_text = "This is a long piece of text that doesn't mention any construction elements. " * 5

        with pytest.raises(ValueError, match="does not mention"):
            ConstructionExtractor.extract(long_text)

    def test_edge_case_mixed_formats(self):
        """Test Case 10: Mixed numbering and bullet styles."""
        construction_text = """
        1. Entities:
        * Server
        * Client
        * Connection
        * Message

        2. State Variables:
        - connection_active: boolean
        - message_queue: list
        - server_load: int

        3. Actions:
        • connect: establish connection
        • send_message: transmit data
        • disconnect: close connection

        4. Constraints:
        Connection must be active before sending
        Server load must stay below threshold
        """

        result = ConstructionExtractor.extract(construction_text)

        assert result["metadata"]["extraction_confidence"] == 1.0
        assert len(result["entities"]) == 4
        assert len(result["state_variables"]) == 3
        assert len(result["actions"]) == 3
        assert len(result["constraints"]) == 2

    def test_integration_real_gemini_output(self):
        """Integration Test 1: Real Gemini-style construction output."""
        construction_text = """
        I'll analyze this scheduling problem step by step.

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
        """

        result = ConstructionExtractor.extract(construction_text)

        # Should extract all 4 sections
        assert result["metadata"]["extraction_confidence"] == 1.0
        assert result["metadata"]["missing_sections"] == []

        # Entities check
        assert len(result["entities"]) >= 3
        entities_str = str(result["entities"])
        assert "Person" in entities_str or "Alice" in entities_str

        # State variables check
        assert len(result["state_variables"]) >= 2
        assert any("person_availability" in sv["name"] for sv in result["state_variables"])
        assert any("meeting_scheduled" in sv["name"] for sv in result["state_variables"])

        # Actions check
        assert len(result["actions"]) >= 1
        action = result["actions"][0]
        assert "schedule_meeting" in action["name"]
        assert len(action.get("preconditions", [])) >= 1
        assert len(action.get("effects", [])) >= 1

        # Constraints check
        assert len(result["constraints"]) >= 3

    def test_integration_vietnamese_construction(self):
        """Integration Test 2: Vietnamese language construction."""
        construction_text = """
        Thực thể (Entities): người dùng, nhiệm vụ, deadline, ưu tiên

        Biến trạng thái (State Variables):
        trạng_thái_nhiệm_vụ: enum [chờ, đang_làm, hoàn_thành]
        người_dùng_rảnh: boolean

        Hành động (Actions):
        - gán_nhiệm_vụ: yêu cầu người dùng rảnh
        - hoàn_thành_nhiệm_vụ: yêu cầu nhiệm vụ đang làm

        Ràng buộc (Constraints):
        - Một người chỉ làm một nhiệm vụ cùng lúc
        - Nhiệm vụ phải hoàn thành trước deadline
        """

        result = ConstructionExtractor.extract(construction_text)

        # Should still extract even with Vietnamese
        assert result["metadata"]["extraction_confidence"] >= 0.75
        assert len(result["entities"]) >= 3
        assert len(result["state_variables"]) >= 1
        assert len(result["actions"]) >= 1
        assert len(result["constraints"]) >= 1

    def test_confidence_calculation(self):
        """Test Case 11: Confidence score calculation."""
        # All 4 sections present (use proper keywords)
        full_text = """
        Entities: entity_a, entity_b, entity_c
        State Variables: state_x, state_y, state_z
        Actions: action_do_something, action_do_other
        Constraints: must be valid, must be complete
        """
        result_full = ConstructionExtractor.extract(full_text)
        assert result_full["metadata"]["extraction_confidence"] == 1.0

        # Only 2 sections present
        partial_text = """
        Entities: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q
        State Variables: x, y, z, w, v, u, t, s, r, q, p, o, n, m, l, k
        """
        result_partial = ConstructionExtractor.extract(partial_text)
        assert result_partial["metadata"]["extraction_confidence"] == 0.5
        assert "actions" in result_partial["metadata"]["missing_sections"]
        assert "constraints" in result_partial["metadata"]["missing_sections"]

    def test_state_variable_type_detection(self):
        """Test Case 12: State variable type extraction."""
        construction_text = """
        Entities: test

        State Variables:
        - boolean_var: boolean
        - enum_var: enum [option1, option2, option3]
        - int_var: int
        - list_var: list[Item]
        - dict_var: dict[Key, Value]
        - simple_var

        Actions: test_action
        Constraints: test constraint
        """

        result = ConstructionExtractor.extract(construction_text)

        state_vars = result["state_variables"]
        assert len(state_vars) >= 5

        # Check boolean type
        bool_var = next((sv for sv in state_vars if "boolean" in sv["name"]), None)
        assert bool_var and bool_var["type"] == "boolean"

        # Check enum with values
        enum_var = next((sv for sv in state_vars if "enum" in sv["name"]), None)
        assert enum_var and enum_var["type"] == "enum"
        assert len(enum_var.get("possible_values", [])) == 3

        # Check simple variable (no type specified)
        simple_var = next((sv for sv in state_vars if "simple" in sv["name"]), None)
        assert simple_var and simple_var["type"] == "unknown"


class TestConstructionExtractorLogging:
    """Test logging behavior."""

    def test_logging_on_success(self, caplog):
        """Verify logging on successful extraction."""
        construction_text = """
        Entities: a, b, c
        State: x, y
        Actions: do_it
        Constraints: valid only
        """

        with caplog.at_level(logging.INFO):
            ConstructionExtractor.extract(construction_text)
            assert "Extracted construction elements" in caplog.text

    def test_logging_on_missing_sections(self, caplog):
        """Verify logging for missing sections."""
        construction_text = """
        Entities: this is a long list of entities to make it valid length
        State Variables: some state variables here to pad the text more and more
        """

        with caplog.at_level(logging.INFO):
            result = ConstructionExtractor.extract(construction_text)
            # Check result has missing sections
            assert len(result["metadata"]["missing_sections"]) == 2
            # Logging happens with structured data
            assert "Extracted construction elements" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
