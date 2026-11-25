"""SageMaker Batch Transform HTTP contract, served by a single Gunicorn worker."""

import json
import uuid
from pathlib import Path

from flask import Flask, Response, request
from werkzeug.exceptions import HTTPException

from online_retail.logging import configure_logging
from online_retail.sagemaker.batch import CONTENT_TYPE, MAX_BYTES, decode_records, validate_records
from online_retail.serving.bundle import load_bundle
from online_retail.serving.inference import predict_features


def create_app(model_dir=Path("/opt/ml/model")):
    logger = configure_logging()
    # Invalid bundles fail worker startup, rather than reporting healthy.
    model, policy, manifest = load_bundle(model_dir)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
    run_id = uuid.uuid4().hex
    logger.info(
        "sagemaker_model_loaded", extra={"context": {"model_version": manifest["model_version"]}}
    )

    @app.route("/ping", methods=["GET", "POST"])
    def ping():
        return Response(status=200)

    @app.get("/execution-parameters")
    def parameters():
        return {"MaxConcurrentTransforms": 1, "BatchStrategy": "MULTI_RECORD", "MaxPayloadInMB": 6}

    @app.post("/invocations")
    def invoke():
        if request.mimetype not in {CONTENT_TYPE, "application/x-ndjson"}:
            return {"error": "Use application/jsonlines"}, 415
        if request.headers.get("Accept") not in {None, "*/*", CONTENT_TYPE, "application/x-ndjson"}:
            return {"error": "Use application/jsonlines response format"}, 406
        try:
            frame = validate_records(decode_records(request.get_data()), policy, manifest)
            scores = predict_features(model, policy, frame)
            for name in ("model_version", "policy_version", "feature_version"):
                scores[name] = manifest[name]
            scores["data_version"] = frame.data_version.to_numpy()
            scores["run_id"] = run_id
            logger.info(
                "sagemaker_batch_scored", extra={"context": {"rows": len(scores), "run_id": run_id}}
            )
            return Response(
                scores.to_json(orient="records", lines=True, date_format="iso"),
                mimetype=CONTENT_TYPE,
            )
        except (ValueError, TypeError, KeyError):
            logger.warning("sagemaker_request_rejected")
            return {"error": "Invalid snapshot records; check schema, versions and dates"}, 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return Response(
            json.dumps({"error": error.name}), status=error.code, mimetype="application/json"
        )

    @app.errorhandler(Exception)
    def unexpected(error):
        logger.exception("sagemaker_invocation_failed")
        return {"error": "Prediction failed; inspect job logs"}, 500

    return app
