from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from model_router.app import mode_weights
from model_router.catalog import add_user_model, catalog_fingerprint, load_catalog, load_catalogs
from model_router.classifier import parse_route


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "coding_llm_models.json"


class RouterBasicsTest(unittest.TestCase):
    def test_catalog_loads_every_model(self):
        profiles = load_catalog(CATALOG)
        self.assertGreater(len(profiles), 10)
        self.assertEqual(len({profile.catalog_key for profile in profiles}), len(profiles))
        self.assertTrue(any(profile.advanced for profile in profiles))

    def test_catalog_fingerprint_changes_with_embedding_model(self):
        first = catalog_fingerprint(CATALOG, "Qwen/Qwen3-Embedding-0.6B")
        second = catalog_fingerprint(CATALOG, "another-model")
        self.assertNotEqual(first, second)

    def test_classifier_parses_json_and_model_style_dictionary(self):
        self.assertEqual(parse_route('{"route": "bug_fixing"}'), "bug_fixing")
        self.assertEqual(parse_route("{'route': 'bug_fixing'}"), "bug_fixing")

    def test_user_model_can_be_added_and_used_as_its_own_candidate_pool(self):
        with TemporaryDirectory() as temporary_directory:
            user_catalog = Path(temporary_directory) / "user_models.json"
            add_user_model(
                user_catalog,
                catalog_key="team-local-coder",
                model="team-coder:14b",
                service="OpenAI-compatible",
                api_endpoint="http://localhost:9000/v1",
                feature="Local coding model for debugging and Python code generation",
                size="14B",
                input_price=0,
                output_price=0,
            )
            profiles = load_catalogs(CATALOG, user_catalog, candidate_pool="user")
            self.assertEqual([profile.catalog_key for profile in profiles], ["team-local-coder"])
            self.assertTrue(profiles[0].user_defined)

    def test_modes_have_explicit_quality_and_cost_policy(self):
        # Paper's five sweep points (Table 2 / Figure 5)
        self.assertEqual(mode_weights("skill_based"), (1.0, 0.0))
        self.assertEqual(mode_weights("quality_leaning"), (0.8, 0.2))
        self.assertEqual(mode_weights("mixed"), (0.6, 0.4))
        self.assertEqual(mode_weights("cost_sensitive"), (0.4, 0.6))
        self.assertEqual(mode_weights("cost_efficient"), (0.2, 0.8))

    def test_custom_alpha_beta_overrides(self):
        # Custom overrides take precedence over named mode defaults
        self.assertEqual(mode_weights("mixed", alpha_override=0.3, beta_override=0.7), (0.3, 0.7))
        # Partial override: only alpha
        alpha, beta = mode_weights("skill_based", alpha_override=0.5)
        self.assertEqual(alpha, 0.5)
        self.assertEqual(beta, 0.0)  # β stays at mode default

    def test_unsupported_mode_raises(self):
        with self.assertRaises(ValueError):
            mode_weights("nonexistent_mode")


if __name__ == "__main__":
    unittest.main()
