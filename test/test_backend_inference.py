"""Smoke test for Model 1 inference through the backend ingestion path."""

import numpy as np

from backend import server


def test_backend_generates_activity_prediction_after_20_samples():
    server.activity_engine.reset()
    server.latest_prediction = None

    sample = np.zeros(6, dtype=float)
    for _ in range(19):
        response = server.receive_imu({
            "ax": sample[0],
            "ay": sample[1],
            "az": sample[2],
            "gx": sample[3],
            "gy": sample[4],
            "gz": sample[5],
        })
        assert response == {"status": "received"}
        assert server.latest_prediction is None

    server.receive_imu({
        "ax": sample[0],
        "ay": sample[1],
        "az": sample[2],
        "gx": sample[3],
        "gy": sample[4],
        "gz": sample[5],
    })

    prediction = server.get_latest()["prediction"]
    assert set(prediction) == {
        "activity",
        "activity_id",
        "confidence",
        "probabilities",
    }
    assert prediction["activity_id"] in range(1, 7)
    assert len(prediction["probabilities"]) == 6
