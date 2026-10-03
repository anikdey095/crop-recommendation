"""
Automated Production Test Suite for Crop Recommendation System
Tests model artifacts, Flask endpoints, dual-engine predictions, error handling,
and latency benchmarks.
"""

import unittest
import json
import os
import time
from api.index import app, predict_pure_python, FEATURE_NAMES

class CropRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.app = app

    def test_01_artifacts_exist(self):
        """Ensure all required production model and metadata files exist."""
        required = ['model_data.json', 'crop_metadata.json', 'model_metrics.json']
        for fname in required:
            path_root = os.path.join(os.path.dirname(__file__), fname)
            path_api = os.path.join(os.path.dirname(__file__), 'api', fname)
            self.assertTrue(os.path.exists(path_root) or os.path.exists(path_api), f"Missing artifact: {fname}")

    def test_02_health_endpoint(self):
        """Test /api/health endpoint returns 200 and healthy status."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get('status'), 'healthy')
        self.assertTrue(data.get('model_loaded'))
        self.assertEqual(data.get('crops_count'), 22)

    def test_03_model_info_endpoint(self):
        """Test /api/model-info returns 99.55% accuracy and feature importances."""
        response = self.client.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('model_metrics', data)
        self.assertGreaterEqual(data['model_metrics']['test_accuracy'], 98.0)
        self.assertEqual(data['total_crops'], 22)
        self.assertIn('rainfall', data['model_metrics']['feature_importances'])

    def test_04_presets_endpoint(self):
        """Test /api/presets returns preset scenarios."""
        response = self.client.get('/api/presets')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 5)
        for preset in data:
            for feat in FEATURE_NAMES:
                self.assertIn(feat, preset)

    def test_05_crops_catalog_endpoint(self):
        """Test /api/crops returns list of 22 crops and detailed info."""
        response = self.client.get('/api/crops')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(len(data), 22)
        
        # Test specific crop lookup
        crop_res = self.client.get('/api/crops/rice')
        self.assertEqual(crop_res.status_code, 200)
        crop_data = crop_res.get_json()
        self.assertEqual(crop_data['name'], 'rice')
        self.assertIn('stats', crop_data)
        self.assertIn('N', crop_data['stats'])

    def test_06_predict_rice(self):
        """Test prediction for Rice scenario."""
        payload = {
            'N': 90, 'P': 42, 'K': 43,
            'temperature': 20.87, 'humidity': 82.0,
            'ph': 6.5, 'rainfall': 202.9
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['prediction']['crop'], 'rice')
        self.assertGreater(data['prediction']['confidence'], 50.0)
        self.assertIn('suitability_analysis', data)
        self.assertEqual(len(data['suitability_analysis']), 7)

    def test_07_predict_coffee(self):
        """Test prediction for Coffee scenario."""
        payload = {
            'N': 105, 'P': 28, 'K': 30,
            'temperature': 25.5, 'humidity': 58.0,
            'ph': 6.8, 'rainfall': 160.0
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['prediction']['crop'], 'coffee')

    def test_08_predict_missing_parameter(self):
        """Test validation when required parameters are missing."""
        payload = {'N': 90, 'P': 42}  # missing other 5
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_09_predict_invalid_data_type(self):
        """Test validation when parameter is not numeric."""
        payload = {
            'N': 'invalid_string', 'P': 42, 'K': 43,
            'temperature': 20.8, 'humidity': 82.0,
            'ph': 6.5, 'rainfall': 200.0
        }
        response = self.client.post('/api/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_10_pure_python_engine_latency(self):
        """Benchmark zero-dependency pure Python tree evaluator."""
        sample = [90, 42, 43, 20.87, 82.0, 6.5, 202.9]
        start = time.time()
        for _ in range(50):
            res = predict_pure_python(sample)
        elapsed_ms = (time.time() - start) * 1000 / 50
        print(f"\n[Benchmark] Pure Python RF inference latency: {elapsed_ms:.2f} ms / prediction")
        self.assertLess(elapsed_ms, 25.0, "Pure python inference should be ultra-fast (<25ms)")
        self.assertEqual(res[0][0], 'rice')

if __name__ == '__main__':
    unittest.main()
