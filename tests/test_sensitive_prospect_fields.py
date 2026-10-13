import json
import os
import unittest
from unittest.mock import patch

os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


SENSITIVE_VALUES = {
    "tax_id": "123-45-6789",
    "date_of_birth": "1980-01-02",
    "card_on_file": "4111111111111111",
    "credit_check_ref": "CHECK-123",
}
PROSPECT_RECORD = {
    "name": "Taylor Example",
    "email": "taylor@example.com",
    "billing_qualification": SENSITIVE_VALUES,
    "annual_revenue": 1000000,
    "tech_stack": ["Python"],
    "account_details": [{"account": "Example Co"}],
    "engagement_history": [{"channel": "Demo"}],
}
OFFERING = {
    "required_tech_stack": ["Python"],
    "min_annual_revenue": 100,
    "description": "Example offering",
}


class SensitiveProspectFieldsTest(unittest.TestCase):
    def assert_no_sensitive_data(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive_value in SENSITIVE_VALUES.values():
            self.assertNotIn(sensitive_value, serialized)

    def test_get_prospect_returns_only_contact_fields(self):
        with patch.object(data_service, "get_prospect_record", return_value=PROSPECT_RECORD):
            result = gtm_agent.get_prospect.func("LEAD-1")

        self.assertEqual(result, {
            "prospect": {
                "prospect_id": "LEAD-1",
                "name": "Taylor Example",
                "email": "taylor@example.com",
            },
            "found": True,
        })
        self.assert_no_sensitive_data(result)

    def test_prospect_not_found_behavior_is_preserved(self):
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=None),
        ):
            self.assertEqual(
                gtm_agent.get_prospect.func("MISSING"),
                {"prospect": None, "found": False},
            )
            self.assertEqual(
                gtm_agent.build_prospect_profile.func("MISSING"),
                {"prospect_profile": None, "found": False},
            )

    def test_build_prospect_profile_sanitizes_new_and_cached_profiles(self):
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=PROSPECT_RECORD),
            patch.object(data_service, "fetch_engagement_history", return_value=PROSPECT_RECORD["engagement_history"]),
            patch.object(data_service, "fetch_account_details", return_value=PROSPECT_RECORD["account_details"]),
            patch.object(data_service, "fetch_tech_stack", return_value=PROSPECT_RECORD["tech_stack"]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            new_result = gtm_agent.build_prospect_profile.func("LEAD-1")

        self.assertEqual(new_result["prospect_profile"]["annual_revenue"], 1000000)
        self.assert_no_sensitive_data(new_result)
        self.assert_no_sensitive_data(save_profile.call_args.args[1])

        cached = {"prospect_id": "LEAD-1", **PROSPECT_RECORD}
        with patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": cached}):
            cached_result = gtm_agent.build_prospect_profile.func("LEAD-1")

        self.assert_no_sensitive_data(cached_result)
        self.assertIn("billing_qualification", cached)

    def test_traceable_cache_lookup_returns_sanitized_copy(self):
        profile = {"prospect_id": "LEAD-1", **PROSPECT_RECORD}
        with patch.dict(data_service._PROFILES, {"LEAD-1": profile}, clear=True):
            result = data_service.get_profile_from_db("LEAD-1")
            self.assertIn("billing_qualification", data_service._PROFILES["LEAD-1"])

        self.assert_no_sensitive_data(result)

    def test_scoring_serializes_only_required_profile_fields(self):
        class ScoringResult:
            def model_dump(self):
                return {"score": 90}

        class ScoringModel:
            def invoke(self, messages):
                self.messages = messages
                return ScoringResult()

        scoring_model = ScoringModel()
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Python"]),
            patch.object(gtm_agent, "_scoring_llm", scoring_model),
        ):
            result = gtm_agent.score_prospect.func(
                {"prospect_id": "LEAD-1", **PROSPECT_RECORD}, OFFERING
            )

        self.assertEqual(result, {"score": 90})
        serialized_input = scoring_model.messages[1]["content"]
        self.assert_no_sensitive_data(serialized_input)
        self.assertIn('"annual_revenue": 1000000', serialized_input)
        self.assertIn('"tech_stack": [\n    "Python"\n  ]', serialized_input)

    def test_email_contact_output_remains_usable(self):
        with patch.object(data_service, "get_prospect_record", return_value=PROSPECT_RECORD):
            contact = gtm_agent.get_prospect.func("LEAD-1")["prospect"]
        email = gtm_agent.send_prospect_email.func(
            contact, "Hello", "A clean message", runtime=None,
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )

        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["to"], "taylor@example.com")
        self.assertEqual(email["to_name"], "Taylor Example")
        self.assert_no_sensitive_data(email)


if __name__ == "__main__":
    unittest.main()
