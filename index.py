import copy
import hashlib
import json
import math
import re
import unicodedata
from collections import deque
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()
_DEBUG_LOGS = deque(maxlen=500)
_DEBUG_SEQUENCE = 0


@app.middleware("http")
async def capture_debug_log(request: Request, call_next):
    global _DEBUG_SEQUENCE
    if request.url.path in ("/ga8/debug-logs", "/ga8/debug-logs/clear"):
        return await call_next(request)
    body = await request.body()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = receive
    response = await call_next(request)
    response_body = b"".join([chunk async for chunk in response.body_iterator])
    if request.url.path.startswith("/ga8/"):
        _DEBUG_SEQUENCE += 1
        try:
            request_data = json.loads(body)
        except Exception:
            request_data = body.decode("utf-8", "replace")
        try:
            response_data = json.loads(response_body)
        except Exception:
            response_data = response_body.decode("utf-8", "replace")
        _DEBUG_LOGS.append({"sequence": _DEBUG_SEQUENCE, "method": request.method, "path": request.url.path,
                            "status": response.status_code, "request": request_data, "response": response_data})
    return Response(response_body, status_code=response.status_code, headers=dict(response.headers))


@app.get("/ga8/debug-logs")
async def debug_logs():
    return {"logs": list(_DEBUG_LOGS)}


@app.post("/ga8/debug-logs/clear")
async def clear_debug_logs():
    _DEBUG_LOGS.clear()
    return {"cleared": True}


_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$")
_GENERATION = re.compile(r"^[0-9]+$")
_URI = re.compile(r"^gs://[^/]+/.+$")
_HEX = re.compile(r"^[0-9a-f]{8}$")
_SAFE = 2**53 - 1


def _time(value):
    if not isinstance(value, str):
        return None
    match = _TIME.fullmatch(value)
    if not match:
        return None
    fraction = (match.group(2) or "").ljust(3, "0")
    offset = match.group(3)
    if offset == "Z":
        tz = timezone.utc
    else:
        hours, minutes = map(int, offset[1:].split(":"))
        if hours > 14 or (hours == 14 and minutes != 0) or minutes > 59:
            return None
        delta = timedelta(hours=hours, minutes=minutes)
        tz = timezone(delta if offset[0] == "+" else -delta)
    try:
        result = datetime.strptime(match.group(1) + "." + fraction, "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=tz)
        result = result.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None
    return f"{result.year:04d}-{result.month:02d}-{result.day:02d}T{result.hour:02d}:{result.minute:02d}:{result.second:02d}.{fraction}Z"


def _crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return f"{crc ^ 0xFFFFFFFF:08x}"


def _canonical(value):
    return " ".join(unicodedata.normalize("NFKC", value).lower().strip().split())


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json(value):
    def invalid_constant(_):
        raise ValueError
    return json.loads(value, parse_constant=invalid_constant)


def _code_list(codes):
    return sorted(set(codes), key=lambda code: code.encode())


def _words(value):
    words, current = set(), []
    for char in value.lower():
        if unicodedata.category(char)[0] in "LN":
            current.append(char)
        elif current:
            words.add("".join(current))
            current = []
    if current:
        words.add("".join(current))
    return words


def _similarity(left, right):
    left, right = _words(left), _words(right)
    if not left and not right:
        return 1
    return len(left & right) / len(left | right)


def _row_json(row):
    return {"id": row["id"], "entity": row["entity"], "eventTime": row["eventTime"], "revision": row["revision"], "text": row["text"]}


def _sort_rows(rows):
    return sorted(rows, key=lambda row: (row["id"].encode(), _compact(_row_json(row)).encode()))


def _empty_result():
    return {"splits": {"train": [], "validation": [], "test": []}, "rejectedObjects": [], "rejectedRows": [], "digests": {}, "lineage": []}

@app.post("/build-corpus")
@app.post("/ga8/build-corpus")
async def build_corpus(request: Request):
    try:
        payload = _json((await request.body()).decode("utf-8"))
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if not isinstance(payload, dict) or "policy" not in payload or not isinstance(payload.get("objects"), list):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    policy = payload["policy"]
    policy_valid = isinstance(policy, dict)
    if policy_valid:
        minimum = _time(policy.get("minTime"))
        maximum = _time(policy.get("maxTime"))
        threshold = policy.get("contaminationThreshold")
        policy_valid = (minimum is not None and maximum is not None and minimum <= maximum
                        and isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
                        and math.isfinite(threshold) and 0 <= threshold <= 1)
    else:
        minimum = maximum = threshold = None

    result = _empty_result()
    candidates = []
    for obj in payload["objects"]:
        obj = obj if isinstance(obj, dict) else {}
        uri = obj.get("uri")
        codes = []
        if not isinstance(uri, str) or not _URI.fullmatch(uri):
            codes.append("URI_INVALID")
        generation = obj.get("generation")
        fetched = obj.get("fetchedGeneration")
        generation_ok = isinstance(generation, str) and _GENERATION.fullmatch(generation) is not None
        fetched_ok = isinstance(fetched, str) and _GENERATION.fullmatch(fetched) is not None
        if not generation_ok or not fetched_ok:
            codes.append("GENERATION_INVALID")
        if generation != fetched:
            codes.append("GENERATION_MISMATCH")
        crc = obj.get("crc32c")
        crc_ok = isinstance(crc, str) and _HEX.fullmatch(crc) is not None
        if not crc_ok:
            codes.append("CRC32C_INVALID")
        content = obj.get("content")
        if not isinstance(content, str):
            codes.append("SCHEMA_INVALID")
        elif crc_ok and _crc32c(content.encode("utf-8")) != crc:
            codes.append("CRC32C_MISMATCH")
        if obj.get("schemaId") != "training-v1":
            codes.append("SCHEMA_INVALID")

        rows = []
        if isinstance(content, str):
            lines = [line for line in content.split("\n") if line.strip()]
            if not lines:
                codes.append("SCHEMA_INVALID")
            else:
                for line in lines:
                    try:
                        raw = _json(line)
                    except Exception:
                        codes.append("JSONL_INVALID")
                        continue
                    if not isinstance(raw, dict) or set(raw) != {"id", "entity", "eventTime", "revision", "text"}:
                        codes.append("SCHEMA_INVALID")
                        continue
                    if (not all(isinstance(raw[key], str) for key in ("id", "entity", "eventTime", "text"))
                            or isinstance(raw["revision"], bool) or not isinstance(raw["revision"], int)
                            or raw["revision"] < 0 or raw["revision"] > _SAFE or _time(raw["eventTime"]) is None):
                        codes.append("SCHEMA_INVALID")
                        continue
                    rows.append({"id": raw["id"], "entity": _canonical(raw["entity"]),
                                 "eventTime": _time(raw["eventTime"]), "revision": raw["revision"],
                                 "text": _canonical(raw["text"])})
        if codes:
            result["rejectedObjects"].append({"uri": uri if isinstance(uri, str) else None, "reasonCodes": _code_list(codes)})
            continue
        candidates.extend(rows)
        result["lineage"].append({"uri": uri, "generation": generation, "crc32c": crc, "schemaId": obj["schemaId"]})

    winners = {}
    for row in candidates:
        key = (row["entity"], row["eventTime"], row["text"])
        previous = winners.get(key)
        if previous is None or (row["revision"], -1) > (previous["revision"], -1) or (row["revision"] == previous["revision"] and row["id"].encode() < previous["id"].encode()):
            if previous is not None:
                result["rejectedRows"].append({"id": previous["id"], "reasonCodes": ["DUPLICATE"]})
            winners[key] = row
        else:
            result["rejectedRows"].append({"id": row["id"], "reasonCodes": ["DUPLICATE"]})

    retained = []
    for row in winners.values():
        reasons = []
        if not policy_valid:
            reasons.append("POLICY_INVALID")
        elif not minimum <= row["eventTime"] <= maximum:
            reasons.append("OUT_OF_WINDOW")
        if reasons:
            result["rejectedRows"].append({"id": row["id"], "reasonCodes": _code_list(reasons)})
        else:
            retained.append(row)

    trains = [row for row in retained if hashlib.sha256(row["entity"].encode()).digest()[0] % 10 < 6]
    for row in retained:
        bucket = hashlib.sha256(row["entity"].encode()).digest()[0] % 10
        if bucket >= 6 and any(_similarity(row["text"], train["text"]) >= threshold for train in trains):
            result["rejectedRows"].append({"id": row["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
        else:
            result["splits"]["train" if bucket < 6 else "validation" if bucket < 8 else "test"].append(row)

    for split in result["splits"]:
        result["splits"][split] = [_row_json(row) for row in _sort_rows(result["splits"][split])]
        body = "".join(_compact(row) + "\n" for row in result["splits"][split]).encode("utf-8")
        result["digests"][split] = hashlib.sha256(body).hexdigest()
    result["rejectedObjects"].sort(key=lambda x: ((x["uri"] or "").encode(), _compact(x).encode()))
    result["rejectedRows"].sort(key=lambda x: (x["id"].encode(), _compact(x).encode()))
    result["lineage"].sort(key=lambda x: (x["uri"].encode(), _compact(x).encode()))
    return JSONResponse(result)


_RUNS = {}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _safe_int(value, positive=False):
    return (isinstance(value, int) and not isinstance(value, bool) and
            (0 < value <= _SAFE if positive else 0 <= value <= _SAFE))


def _selection_response(run_id, selected=None, train=None, evaluate=None, features=None, digest=None, codes=None):
    return {"runId": run_id, "selectedTrialId": selected, "trainRowIds": train or [],
            "evalRowIds": evaluate or [], "featureNames": features or [],
            "datasetDigest": digest, "reasonCodes": _code_list(codes or [])}


def _select(data):
    run_id = data.get("runId")
    valid_run = isinstance(run_id, str) and 0 < len(run_id) <= 128
    rows, trials = data.get("rows"), data.get("trials")
    forbidden, limit = data.get("forbiddenFeatures"), data.get("numTrialsLimit")
    valid = (valid_run and isinstance(rows, list) and bool(rows) and isinstance(trials, list)
             and isinstance(forbidden, list) and all(isinstance(x, str) for x in forbidden)
             and _safe_int(limit, positive=True))
    parsed = []
    row_ids = set()
    if isinstance(rows, list):
        for row in rows:
            row_ok = isinstance(row, dict)
            if row_ok:
                row_id, entity = row.get("id"), row.get("entity")
                event, prediction = _time(row.get("eventTime")), _time(row.get("predictionTime"))
                version, split, feature_map = row.get("version"), row.get("split"), row.get("features")
                row_ok = (isinstance(row_id, str) and isinstance(entity, str) and event is not None
                          and prediction is not None and _safe_int(version) and split in ("TRAIN", "EVAL")
                          and isinstance(feature_map, dict) and row_id not in row_ids)
                checked_features = {}
                if row_ok:
                    row_ids.add(row_id)
                    for name, feature in feature_map.items():
                        if (not isinstance(name, str) or not isinstance(feature, dict)
                                or "value" not in feature or _time(feature.get("availableAt")) is None):
                            row_ok = False
                            break
                        checked_features[name] = _time(feature["availableAt"])
            if not row_ok:
                valid = False
            else:
                parsed.append({"id": row_id, "entity": entity, "eventTime": event, "predictionTime": prediction,
                               "version": version, "split": split, "features": checked_features})
    trial_ids, eligible_trials = set(), []
    if isinstance(trials, list):
        for trial in trials:
            trial_ok = (isinstance(trial, dict) and _safe_int(trial.get("trialId"))
                        and trial.get("status") in ("SUCCEEDED", "FAILED")
                        and trial.get("trialId") not in trial_ids)
            if not trial_ok:
                valid = False
                continue
            trial_ids.add(trial["trialId"])
            metric = trial.get("evalMetric")
            if trial["status"] == "SUCCEEDED" and isinstance(metric, (int, float)) and not isinstance(metric, bool) and math.isfinite(metric):
                eligible_trials.append((metric, trial["trialId"]))

    codes = []
    if not valid:
        codes.append("INVALID_INPUT")
    if isinstance(trials, list) and _safe_int(limit, positive=True) and len(trials) > limit:
        codes.append("TRIAL_LIMIT_EXCEEDED")
    if not eligible_trials:
        codes.append("NO_SUCCESSFUL_TRIAL")
    if not valid:
        return _selection_response(run_id, codes=codes)

    retained = {}
    for row in parsed:
        key = (row["entity"], row["eventTime"])
        old = retained.get(key)
        if old is None or row["version"] > old["version"] or (row["version"] == old["version"] and row["id"].encode() < old["id"].encode()):
            retained[key] = row
    retained = list(retained.values())
    names = set.intersection(*(set(row["features"]) for row in retained)) if retained else set()
    names = sorted((name for name in names if name not in set(forbidden)
                    and all(row["features"][name] <= row["predictionTime"] for row in retained)), key=lambda x: x.encode())
    train = sorted((row["id"] for row in retained if row["split"] == "TRAIN"), key=lambda x: x.encode())
    evaluate = sorted((row["id"] for row in retained if row["split"] == "EVAL"), key=lambda x: x.encode())
    artifact = {"trainRowIds": train, "evalRowIds": evaluate, "featureNames": names}
    digest = hashlib.sha256(_compact(artifact).encode()).hexdigest()
    selected = min(eligible_trials, key=lambda x: (-x[0], x[1]))[1] if eligible_trials else None
    if codes:
        selected = None
    return _selection_response(run_id, selected, train, evaluate, names, digest, codes)


def _evaluation_response(data, metric=None, slice_pass=False, decision="reject", codes=None):
    return {"runId": data.get("runId"), "selectedTrialId": data.get("selectedTrialId"),
            "datasetDigest": data.get("datasetDigest"), "testMetric": metric,
            "criticalSlicePass": slice_pass, "decision": decision,
            "bytesProcessed": data.get("bytesProcessed"), "reasonCodes": _code_list(codes or [])}


def _evaluate(data):
    run_id, selected, digest = data.get("runId"), data.get("selectedTrialId"), data.get("datasetDigest")
    floor, required, rows = data.get("metricFloor"), data.get("requiredSlices"), data.get("rows")
    processed, maximum = data.get("bytesProcessed"), data.get("maxBytes")
    finite_floor = isinstance(floor, (int, float)) and not isinstance(floor, bool) and math.isfinite(floor) and 0 <= floor <= 1
    required_ok = isinstance(required, dict) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1
        for value in required.values())
    bytes_ok = _safe_int(processed) and _safe_int(maximum)
    lineage_shape = isinstance(run_id, str) and _safe_int(selected) and isinstance(digest, str) and bool(_HEX64.fullmatch(digest))
    valid = finite_floor and required_ok and isinstance(rows, list) and bytes_ok
    codes = []
    if not valid:
        codes.append("INVALID_INPUT")
    stored = _RUNS.get(run_id, {}).get("response")
    lineage_ok = (lineage_shape and stored is not None and not stored["reasonCodes"]
                  and selected == stored["selectedTrialId"] and digest == stored["datasetDigest"])
    if not lineage_ok:
        codes.append("INVALID_LINEAGE")
    if bytes_ok and processed > maximum:
        codes.append("BYTE_LIMIT")

    test_rows_ok = isinstance(rows, list) and bool(rows)
    counts = {}
    correct = total = 0
    if isinstance(rows, list):
        for row in rows:
            ok = (isinstance(row, dict) and isinstance(row.get("label"), int)
                  and not isinstance(row.get("label"), bool) and row.get("label") in (0, 1)
                  and isinstance(row.get("prediction"), int) and not isinstance(row.get("prediction"), bool)
                  and row.get("prediction") in (0, 1) and isinstance(row.get("slice"), str) and bool(row["slice"]))
            if not ok:
                test_rows_ok = False
                continue
            hit = int(row["label"] == row["prediction"])
            correct += hit
            total += 1
            bucket = counts.setdefault(row["slice"], [0, 0])
            bucket[0] += hit
            bucket[1] += 1
    metric = None
    slice_pass = valid and lineage_ok and test_rows_ok
    if not test_rows_ok:
        codes.append("INVALID_TEST_ROW")
    else:
        metric = round(correct / total, 12)
        if finite_floor and metric < floor:
            codes.append("AGGREGATE_FLOOR")
        if required_ok:
            for name, required_floor in required.items():
                if name not in counts:
                    codes.append("MISSING_SLICE:" + name)
                    slice_pass = False
                elif round(counts[name][0] / counts[name][1], 12) < required_floor:
                    codes.append("SLICE_FLOOR:" + name)
                    slice_pass = False
    decision = "admit" if not codes else "reject"
    return _evaluation_response(data, metric, slice_pass, decision, codes)

@app.post("/bqml")
@app.post("/ga8/bqml")
async def bqml(request: Request):
    try:
        data = _json((await request.body()).decode("utf-8"))
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if not isinstance(data, dict) or data.get("phase") not in ("select", "evaluate"):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if data["phase"] == "evaluate":
        response = _evaluate(data)
        return JSONResponse(response)
    response = _select(data)
    run_id = data.get("runId")
    if isinstance(run_id, str) and 0 < len(run_id) <= 128:
        fingerprint = _compact(data)
        old = _RUNS.get(run_id)
        if old and old["fingerprint"] != fingerprint:
            return JSONResponse({"error": "RUN_ID_CONFLICT"}, status_code=409)
        if old:
            return JSONResponse(old["response"])
        _RUNS[run_id] = {"fingerprint": fingerprint, "response": response}
    return JSONResponse(response)


_CHAMPION_ALIAS = None
_VERSION = re.compile(r"^[1-9][0-9]*$")


def _instant(value):
    normalized = _time(value)
    if normalized is None:
        return None
    return datetime.strptime(normalized, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)


def _number(value, lower=None, upper=None):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return False
    return (lower is None or value >= lower) and (upper is None or value <= upper)


def _promotion_policy(policy, as_of):
    if not isinstance(policy, dict) or as_of is None:
        return False
    required = policy.get("requiredSlices")
    return (isinstance(policy.get("datasetDigest"), str) and bool(policy["datasetDigest"])
            and isinstance(policy.get("schemaDigest"), str) and bool(policy["schemaDigest"])
            and _safe_int(policy.get("maxAgeSeconds"))
            and _number(policy.get("accuracyFloor"), 0, 1)
            and isinstance(required, dict) and all(_number(value, 0, 1) for value in required.values())
            and _number(policy.get("maxLatencyMs"), 0)
            and _safe_int(policy.get("maxSizeBytes"))
            and _number(policy.get("minImprovement"), 0, 1))


def _version_gates(item, policy, as_of, policy_ok):
    codes = []
    if not policy_ok:
        codes.append("INVALID_POLICY")
    evaluation = item.get("evaluation") if isinstance(item, dict) else None
    if not isinstance(evaluation, dict):
        codes.append("MISSING_EVALUATION")
        return _code_list(codes)
    created = _instant(evaluation.get("createdAt"))
    if created is None:
        codes.append("INVALID_TIMESTAMP")
    elif as_of is not None and policy_ok:
        if created > as_of:
            codes.append("FUTURE_EVALUATION")
        elif created < as_of - timedelta(seconds=policy["maxAgeSeconds"]):
            codes.append("STALE_EVALUATION")

    accuracy, latency, size = evaluation.get("accuracy"), evaluation.get("latencyMs"), evaluation.get("sizeBytes")
    accuracy_finite, latency_finite = _number(accuracy), _number(latency)
    size_finite = isinstance(size, (int, float)) and not isinstance(size, bool) and math.isfinite(size)
    if not accuracy_finite or not latency_finite or not size_finite:
        codes.append("NON_FINITE")
    if accuracy_finite and not 0 <= accuracy <= 1:
        codes.append("METRIC_RANGE")
    if latency_finite and latency < 0:
        codes.append("METRIC_RANGE")
    if size_finite and not _safe_int(size):
        codes.append("METRIC_RANGE")

    registered_artifact = item.get("artifactDigest")
    if (not isinstance(registered_artifact, str) or not registered_artifact
            or evaluation.get("artifactDigest") != registered_artifact):
        codes.append("ARTIFACT_MISMATCH")
    if policy_ok:
        if evaluation.get("datasetDigest") != policy["datasetDigest"]:
            codes.append("DATASET_MISMATCH")
        if evaluation.get("schemaDigest") != policy["schemaDigest"]:
            codes.append("SCHEMA_MISMATCH")
        if _number(accuracy) and accuracy < policy["accuracyFloor"]:
            codes.append("ACCURACY_FLOOR")
        if _number(latency) and latency > policy["maxLatencyMs"]:
            codes.append("LATENCY_LIMIT")
        if _safe_int(size) and size > policy["maxSizeBytes"]:
            codes.append("SIZE_LIMIT")
        slices = evaluation.get("slices")
        if isinstance(slices, dict):
            for name, value in slices.items():
                if not _number(value):
                    codes.append("NON_FINITE")
                elif not 0 <= value <= 1:
                    codes.append("SLICE_RANGE:" + name)
        for name, floor in policy["requiredSlices"].items():
            if not isinstance(slices, dict) or name not in slices:
                codes.append("MISSING_SLICE:" + name)
                continue
            value = slices[name]
            if _number(value, 0, 1) and value < floor:
                codes.append("SLICE_FLOOR:" + name)
    return _code_list(codes)

@app.post("/promote")
@app.post("/ga8/promote")
async def promote(request: Request):
    global _CHAMPION_ALIAS
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    if (not isinstance(data, dict) or not isinstance(data.get("policy"), dict)
            or not isinstance(data.get("versions"), list) or not isinstance(data.get("championVersion"), str)):
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)

    versions = data["versions"]
    as_of = _instant(data.get("asOf"))
    policy = data["policy"]
    policy_ok = _promotion_policy(policy, as_of)
    counts = {}
    for item in versions:
        version = item.get("version") if isinstance(item, dict) else None
        key = version if isinstance(version, str) else _compact(version)
        counts[key] = counts.get(key, 0) + 1

    failed, lookup, eligible = {}, {}, []
    for item in versions:
        version = item.get("version") if isinstance(item, dict) else None
        key = version if isinstance(version, str) else _compact(version)
        codes = []
        canonical = isinstance(version, str) and bool(_VERSION.fullmatch(version)) and int(version) <= _SAFE
        if not canonical:
            codes.append("INVALID_VERSION")
        if counts[key] > 1:
            codes.append("DUPLICATE_VERSION")
        if canonical and counts[key] == 1:
            lookup[version] = item
            codes.extend(_version_gates(item, policy, as_of, policy_ok))
            if not codes:
                evaluation = item["evaluation"]
                eligible.append((version, evaluation))
        failed[key] = _code_list(codes)

    effective = data["championVersion"]
    champion_ok = effective in lookup and not failed.get(effective)
    eligible.sort(key=lambda pair: (-pair[1]["accuracy"], pair[1]["latencyMs"], pair[1]["sizeBytes"], int(pair[0])))
    eligible_ids = [pair[0] for pair in eligible]
    action, selected = "block", None
    if champion_ok:
        selected = effective
        challenger = eligible[0][0] if eligible else effective
        improvement = round(lookup[challenger]["evaluation"]["accuracy"] - lookup[effective]["evaluation"]["accuracy"], 12)
        if challenger != effective and improvement >= policy["minImprovement"]:
            action, selected, _CHAMPION_ALIAS = "promote", challenger, challenger
        else:
            action = "retain"
    evidence = lookup[selected]["evaluation"] if selected in lookup else None
    failed = {key: failed[key] for key in sorted(failed, key=lambda x: x.encode())}
    response={"action": action, "championVersion": effective, "selectedVersion": selected,
              "eligibleVersions": eligible_ids, "failedGates": failed,
              "aliasMutation": {"alias": "champion", "version": selected} if action == "promote" else None,
              "evidence": evidence}
    return JSONResponse(response)


_ADAPT_NAMES = ["prompt_only", "retrieval", "lora", "qlora"]
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _adapt_choose(data):
    policy, candidates = data.get("policy"), data.get("candidates")
    valid_policy = isinstance(policy, dict)
    if valid_policy:
        valid_policy = (_number(policy.get("minQuality"), 0, 1) and isinstance(policy.get("freshnessRequired"), bool)
            and _number(policy.get("maxLatencyMs"), 0) and _number(policy.get("maxMemoryMb"), 0)
            and _safe_int(policy.get("maxLabeledExamples")) and _number(policy.get("maxTotalCost"), 0)
            and _safe_int(policy.get("horizonRequests")))
    by_name = {}
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
                by_name.setdefault(candidate["name"], []).append(candidate)
    exact = isinstance(candidates, list) and set(by_name) == set(_ADAPT_NAMES) and all(len(by_name[n]) == 1 for n in _ADAPT_NAMES)
    eligible, costs, reasons = [], {}, {}
    for name in _ADAPT_NAMES:
        codes, cost = [], None
        candidate = by_name.get(name, [None])[0]
        fields_ok = isinstance(candidate, dict) and isinstance(candidate.get("available"), bool) and isinstance(candidate.get("freshness"), bool)
        if fields_ok:
            fields_ok = (_number(candidate.get("quality"), 0, 1) and _number(candidate.get("latencyMs"), 0)
                and _number(candidate.get("memoryMb"), 0) and _safe_int(candidate.get("labeledExamples"))
                and _number(candidate.get("oneTimeCost"), 0) and _number(candidate.get("recurringCost"), 0))
        if not valid_policy or not exact or not fields_ok:
            codes.append("INVALID_INPUT")
        if valid_policy and fields_ok:
            cost = round(candidate["oneTimeCost"] + policy["horizonRequests"] * candidate["recurringCost"], 12)
            if not candidate["available"]: codes.append("UNAVAILABLE")
            if candidate["quality"] < policy["minQuality"]: codes.append("QUALITY_FLOOR")
            if policy["freshnessRequired"] and not candidate["freshness"]: codes.append("FRESHNESS_REQUIRED")
            if candidate["latencyMs"] > policy["maxLatencyMs"]: codes.append("LATENCY_LIMIT")
            if candidate["memoryMb"] > policy["maxMemoryMb"]: codes.append("MEMORY_LIMIT")
            if candidate["labeledExamples"] > policy["maxLabeledExamples"]: codes.append("DATA_LIMIT")
            if cost > policy["maxTotalCost"]: codes.append("COST_LIMIT")
        reasons[name] = _code_list(codes)
        costs[name] = cost
        if not codes: eligible.append(name)
    return {"selected": eligible[0] if eligible else None, "eligible": eligible, "totalCosts": costs, "reasonCodes": reasons}


def _adapt_repair(data):
    codes = []
    tokens = data.get("tokens")
    tokens_ok = isinstance(tokens, list) and bool(tokens)
    labels = []
    if tokens_ok:
        seen = set()
        for token in tokens:
            ok = (isinstance(token, dict) and _safe_int(token.get("id")) and token.get("id") not in seen
                  and token.get("role") in ("system", "user", "assistant") and isinstance(token.get("padding"), bool)
                  and isinstance(token.get("text"), str))
            if not ok: tokens_ok = False
            else: seen.add(token["id"])
        labels = [t["id"] if t["role"] == "assistant" and not t["padding"] else -100 for t in tokens] if tokens_ok else [-100] * len(tokens)
    if not tokens_ok:
        codes.append("INVALID_TOKEN")
        if isinstance(tokens, list): labels = [-100] * len(tokens)
    template_pass = _safe_int(data.get("templateApplications")) and data.get("templateApplications") == 1
    if not template_pass: codes.append("CHAT_TEMPLATE_COUNT")

    params, allowed = data.get("parameters"), data.get("allowedTargets")
    allowed_shape = isinstance(allowed, list) and bool(allowed) and all(isinstance(x, str) and x for x in allowed)
    allowed_eligible = allowed_shape and len(set(allowed)) == len(allowed)
    params_ok = isinstance(params, list) and allowed_eligible
    trainable, count, names = [], 0, set()
    if params_ok:
        for param in params:
            valid_param = (isinstance(param, dict) and isinstance(param.get("name"), str)
                           and isinstance(param.get("target"), str) and _safe_int(param.get("numel"), positive=True))
            if not valid_param or param["name"] in names:
                params_ok = False
                break
            names.add(param["name"])
            if param["target"] in allowed and param["name"].endswith((".lora_A.weight", ".lora_B.weight")):
                trainable.append(param["name"])
                count += param["numel"]
    if not trainable: params_ok = False
    if count > _SAFE: params_ok, trainable, count = False, [], 0
    trainable.sort(key=lambda x: x.encode())
    if not params_ok: codes.append("INVALID_PARAMETER")
    inference_ok = data.get("inferenceMode") is False
    if not inference_ok: codes.append("INFERENCE_MODE")
    peft_pass = params_ok and inference_ok

    files = data.get("artifactFiles")
    adapter_files = ["adapter_config.json", "adapter_model.safetensors"]
    full_model = isinstance(files, list) and any(isinstance(x, str) and (x == "model.safetensors" or x.endswith((".bin", ".pt", ".pth"))) for x in files)
    if full_model: codes.append("FULL_MODEL_ARTIFACT")
    file_pass = sorted(files) == adapter_files if isinstance(files, list) else False
    if not file_pass: codes.append("ADAPTER_FILE_SET")
    peft_pass = params_ok and inference_ok and file_pass and not full_model

    checkpoint = data.get("checkpoint")
    checkpoint_ok = isinstance(checkpoint, dict) and all(k in checkpoint for k in ("model", "optimizer", "scheduler", "step", "rng", "dataPosition"))
    if not checkpoint_ok: codes.append("INCOMPLETE_CHECKPOINT")
    base_ok = isinstance(data.get("baseRevision"), str) and bool(_HEX40.fullmatch(data["baseRevision"]))
    if not base_ok: codes.append("MUTABLE_BASE_REVISION")
    expected = data.get("expectedDigests")
    lineage_ok = isinstance(expected, dict) and base_ok
    for key in ("datasetDigest", "codeDigest", "configDigest"):
        value = data.get(key)
        lineage_ok = lineage_ok and isinstance(value, str) and bool(_HEX64.fullmatch(value)) and expected.get(key) == value
    if not lineage_ok: codes.append("LINEAGE_MISMATCH")
    factors = [data.get(k) for k in ("microBatch", "gradientAccumulation", "replicas", "expectedEffectiveBatch")]
    batch_ok = all(_safe_int(x, positive=True) for x in factors) and factors[0] * factors[1] * factors[2] == factors[3]
    if not batch_ok: codes.append("EFFECTIVE_BATCH_MISMATCH")
    train, evaluate = data.get("trainRowIds"), data.get("evalRowIds")
    eval_ok = (isinstance(train, list) and isinstance(evaluate, list) and bool(train) and bool(evaluate)
               and all(isinstance(x, str) and x for x in train + evaluate) and len(set(train)) == len(train)
               and len(set(evaluate)) == len(evaluate) and set(train).isdisjoint(evaluate))
    if not eval_ok: codes.append("EVAL_LEAKAGE")
    deterministic = data.get("dropoutActiveDuringEval") is False
    if not deterministic: codes.append("EVAL_DROPOUT_ACTIVE")
    left, right, tolerance = data.get("uninterruptedWeights"), data.get("resumedWeights"), data.get("resumeTolerance")
    resume_ok = (isinstance(left, list) and isinstance(right, list) and bool(left) and len(left) == len(right)
                 and _number(tolerance, 0) and all(_number(x) for x in left + right)
                 and all(abs(a-b) <= tolerance for a,b in zip(left,right)))
    if not resume_ok: codes.append("RESUME_DIVERGENCE")
    return {"labels": labels, "templatePass": bool(template_pass), "trainableParams": trainable, "trainableCount": count,
            "peftConfigPass": peft_pass, "adapterFiles": adapter_files, "checkpointComplete": checkpoint_ok,
            "lineagePass": lineage_ok and batch_ok, "evalIsolated": eval_ok, "evaluationDeterministic": deterministic,
            "resumePass": resume_ok, "reasonCodes": _code_list(codes)}


@app.post("/ga8/adapt")
async def adapt(request: Request):
    try: data = await request.json()
    except Exception: return JSONResponse({"error":"INVALID_INPUT"}, status_code=400)
    if not isinstance(data, dict) or data.get("operation") not in ("choose", "repair"):
        return JSONResponse({"error":"INVALID_INPUT"}, status_code=400)
    response = _adapt_choose(data) if data["operation"] == "choose" else _adapt_repair(data)
    return JSONResponse(response)


_FREEZES = {}


def _freeze(data):
    freeze_id, candidates = data.get("freezeId"), data.get("candidates")
    allowed = data.get("allowedUnsupportedReasons")
    top_ok = (isinstance(freeze_id, str) and 0 < len(freeze_id) <= 128
              and isinstance(data.get("calibrationDigest"), str) and bool(data["calibrationDigest"])
              and isinstance(data.get("tokenizerDigest"), str) and bool(data["tokenizerDigest"])
              and isinstance(allowed, list) and all(isinstance(x, str) and x for x in allowed) and len(set(allowed)) == len(allowed))
    names = [x.get("name") for x in candidates if isinstance(x, dict)]
    names_ok = len(names) == len(candidates) and all(isinstance(x, str) and x for x in names) and len(set(names)) == len(names)
    output = []
    for candidate in candidates:
        name = candidate.get("name") if isinstance(candidate, dict) else None
        codes, inventory, total, package = [], [], None, None
        files = candidate.get("files") if isinstance(candidate, dict) else None
        files_ok = isinstance(files, dict) and bool(files) and all(isinstance(k, str) and k and isinstance(v, str) for k,v in files.items())
        candidate_ok = top_ok and names_ok and isinstance(candidate, dict) and isinstance(candidate.get("loadable"), bool) and isinstance(name, str) and bool(name)
        if not candidate_ok or not files_ok:
            codes.append("INVALID_INPUT")
        if files_ok:
            for filename in sorted(files, key=lambda x: x.encode()):
                raw = files[filename].encode()
                inventory.append({"name":filename,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
            total = sum(x["bytes"] for x in inventory)
            package = hashlib.sha256(_compact(inventory).encode()).hexdigest()
        unsupported = candidate.get("unsupportedReason") if isinstance(candidate, dict) else None
        unsupported_allowed = isinstance(unsupported, str) and bool(unsupported) and unsupported in allowed
        if unsupported and not unsupported_allowed:
            codes.append("UNALLOWED_UNSUPPORTED_REASON")
        if not unsupported_allowed:
            if isinstance(candidate, dict) and candidate.get("loadable") is not True: codes.append("NOT_LOADABLE")
            if isinstance(candidate, dict) and candidate.get("calibrationDigest") != data.get("calibrationDigest"): codes.append("CALIBRATION_MISMATCH")
            if isinstance(candidate, dict) and candidate.get("tokenizerDigest") != data.get("tokenizerDigest"): codes.append("TOKENIZER_MISMATCH")
        status = "invalid" if codes else "unsupported" if unsupported_allowed else "frozen"
        output.append({"name":name,"status":status,"inventory":inventory if files_ok else [],"totalBytes":total if files_ok else None,
                       "packageDigest":package if files_ok else None,"reasonCodes":_code_list(codes)})
    output.sort(key=lambda x: x["name"].encode() if isinstance(x["name"],str) else b"")
    return {"freezeId":freeze_id,"candidates":output}


def _quant_select(data):
    freeze_id, supplied, rows, policy = data.get("freezeId"), data.get("candidates"), data.get("rows"), data.get("policy")
    frozen = _FREEZES.get(freeze_id, {}).get("response")
    required = policy.get("requiredSlices") if isinstance(policy,dict) else None
    order = policy.get("candidateOrder") if isinstance(policy,dict) else None
    policy_ok = (isinstance(policy,dict) and _safe_int(policy.get("maxBytes")) and _number(policy.get("aggregateFloor"),0,1)
                 and isinstance(required,dict) and all(_number(x,0,1) for x in required.values())
                 and _number(policy.get("maxLatencyMs"),0) and isinstance(order,list) and all(isinstance(x,str) for x in order)
                 and len(set(order)) == len(order))
    names = [x.get("name") for x in supplied if isinstance(x,dict)]
    sets_ok = policy_ok and len(names)==len(supplied) and set(names)==set(order) and len(set(names))==len(names)
    recorded = {candidate["name"]: candidate for candidate in frozen["candidates"]} if frozen else {}
    lineage_ok = frozen is not None and supplied == frozen["candidates"]
    results=[]
    for position,name in enumerate(order if isinstance(order,list) else names):
        candidate = next((x for x in supplied if isinstance(x,dict) and x.get("name")==name), {})
        codes=[]
        if candidate.get("status") != "frozen": codes.append("NOT_FROZEN")
        if not lineage_ok: codes.append("INVALID_LINEAGE")
        if not policy_ok or not sets_ok: codes.append("INVALID_POLICY")
        inventory=candidate.get("inventory")
        manifest_ok=isinstance(inventory,list) and all(isinstance(x,dict) and list(x)==["name","bytes","sha256"] and isinstance(x["name"],str)
                   and _safe_int(x["bytes"]) and isinstance(x["sha256"],str) and bool(_HEX64.fullmatch(x["sha256"])) for x in inventory)
        manifest_ok = manifest_ok and [x["name"] for x in inventory] == sorted({x["name"] for x in inventory}, key=lambda x:x.encode())
        total=sum(x["bytes"] for x in inventory) if manifest_ok else None
        package=hashlib.sha256(_compact(inventory).encode()).hexdigest() if manifest_ok else None
        manifest_ok = manifest_ok and total==candidate.get("totalBytes") and package==candidate.get("packageDigest")
        if not manifest_ok: codes.append("INVALID_MANIFEST"); total=None
        latency=data.get("latencies",{}).get(name) if isinstance(data.get("latencies"),dict) else None
        latency_ok=_number(latency,0)
        if not latency_ok: codes.append("INVALID_POLICY"); latency=None
        predictions_ok=isinstance(rows,list) and bool(rows)
        counts={}; correct=0
        if isinstance(rows,list):
            for row in rows:
                pred=row.get("predictions",{}).get(name) if isinstance(row,dict) and isinstance(row.get("predictions"),dict) else None
                ok=(isinstance(row,dict) and isinstance(row.get("label"),int) and not isinstance(row.get("label"),bool) and row["label"] in (0,1)
                    and isinstance(row.get("slice"),str) and bool(row["slice"]) and isinstance(pred,int) and not isinstance(pred,bool) and pred in (0,1))
                if not ok: predictions_ok=False; continue
                hit=int(pred==row["label"]); correct+=hit; bucket=counts.setdefault(row["slice"],[0,0]); bucket[0]+=hit; bucket[1]+=1
        aggregate=round(correct/len(rows),12) if predictions_ok else None
        required_names = list(required) if isinstance(required, dict) else []
        slices={k:(round(counts[k][0]/counts[k][1],12) if predictions_ok and k in counts else None) for k in required_names}
        if not predictions_ok: codes.append("INVALID_PREDICTIONS")
        elif policy_ok:
            if aggregate < policy["aggregateFloor"]: codes.append("AGGREGATE_FLOOR")
            for slice_name,floor in required.items():
                if slices[slice_name] is None: codes.append("MISSING_SLICE:"+slice_name)
                elif slices[slice_name] < floor: codes.append("SLICE_FLOOR:"+slice_name)
        if policy_ok and total is not None and total>policy["maxBytes"]: codes.append("SIZE_LIMIT")
        if policy_ok and latency is not None and latency>policy["maxLatencyMs"]: codes.append("LATENCY_LIMIT")
        results.append({"name":name,"aggregate":aggregate,"slices":{k:slices[k] for k in sorted(slices,key=lambda x:x.encode())},
                        "totalBytes":total,"latencyMs":latency,"admitted":not codes,"reasonCodes":_code_list(codes),"_position":position,"_candidate":candidate})
    admitted=[x for x in results if x["admitted"]]
    winner=min(admitted,key=lambda x:(x["totalBytes"],x["latencyMs"],x["_position"],x["name"].encode())) if admitted else None
    manifest=recorded.get(winner["name"]) if winner else None
    for x in results: x.pop("_position"); x.pop("_candidate")
    return {"freezeId":freeze_id,"selected":winner["name"] if winner else None,"results":results,"packageManifest":manifest}


@app.post("/ga8/quantize")
async def quantize(request: Request):
    try:data=await request.json()
    except Exception:return JSONResponse({"error":"INVALID_INPUT"},status_code=400)
    if not isinstance(data,dict) or data.get("phase") not in ("freeze","select"):
        return JSONResponse({"error":"INVALID_INPUT"},status_code=400)
    if data["phase"]=="freeze":
        if not isinstance(data.get("candidates"),list) or not data["candidates"]:
            return JSONResponse({"error":"INVALID_INPUT"},status_code=400)
        response=_freeze(data); fid=data.get("freezeId")
        if isinstance(fid,str) and 0<len(fid)<=128:
            old=_FREEZES.get(fid)
            if old and old["input"] != data:return JSONResponse({"error":"FREEZE_ID_CONFLICT"},status_code=409)
            if old:return JSONResponse(old["response"])
            _FREEZES[fid]={"input":copy.deepcopy(data),"response":response}
        return JSONResponse(response)
    if not isinstance(data.get("candidates"),list) or not isinstance(data.get("rows"),list) or not isinstance(data.get("policy"),dict):
        return JSONResponse({"error":"INVALID_INPUT"},status_code=400)
    response = _quant_select(data)
    return JSONResponse(response)


_PIPELINES = {}
_DAG = ["verify_data","prepare","train","evaluate","register","publish"]
_INPUTS = ["generation","checksum","canonicalData","prepareCode","prepareConfig","trainCode","trainConfig","runtime","evaluateCode","evaluateConfig","schemaDigest","publishConfig"]


def _pipeline_deps(node, inputs, artifacts):
    if node=="verify_data": return {"generation":inputs["generation"],"checksum":inputs["checksum"]}
    if node=="prepare": return {"canonicalData":inputs["canonicalData"],"prepareCode":inputs["prepareCode"],"prepareConfig":inputs["prepareConfig"]}
    if node=="train": return {"prepareArtifact":artifacts.get("prepare"),"trainCode":inputs["trainCode"],"trainConfig":inputs["trainConfig"],"runtime":inputs["runtime"]}
    if node=="evaluate": return {"trainArtifact":artifacts.get("train"),"canonicalData":inputs["canonicalData"],"evaluateCode":inputs["evaluateCode"],"evaluateConfig":inputs["evaluateConfig"]}
    if node=="register": return {"evaluateArtifact":artifacts.get("evaluate"),"schemaDigest":inputs["schemaDigest"]}
    return {"registerArtifact":artifacts.get("register"),"publishConfig":inputs["publishConfig"]}


def _pipeline_view(state, accepted, ignored):
    artifacts={}; nodes=[]; upstream_terminal=False; upstream_pending=False
    for node in _DAG:
        deps=_pipeline_deps(node,state["inputs"],artifacts)
        ready=all(v is not None for v in deps.values()) and (node == "verify_data" or _DAG[_DAG.index(node)-1] in artifacts)
        key=hashlib.sha256(_compact(list(deps.values())).encode()).hexdigest() if ready else None
        dependencies=dict(deps); dependencies["cacheKey"]=key
        cache=state["cache"].get((node,key)) if key else None
        current=state["states"].get(node)
        triggers=[]
        if upstream_terminal:
            action,reason="block","UPSTREAM_TERMINAL"
        elif upstream_pending or not ready:
            action,reason="block","UPSTREAM_PENDING"; upstream_pending=True
        elif cache:
            action,reason="reuse","CACHE_HIT"; triggers=[cache["eventId"]]; artifacts[node]=cache["artifact"]
        elif current and current.get("key")==key and current["status"]=="terminal_failed":
            action,reason="block","TERMINAL_FAILURE"; triggers=[current["eventId"]]; upstream_terminal=True
        elif current and current.get("key")==key and current["status"]=="started":
            action,reason="block","RUNNING"; triggers=[current["eventId"]]; upstream_pending=True
        elif current and current.get("key")==key and current["status"]=="retryable_failed":
            action,reason="rerun","RETRYABLE_FAILURE"; triggers=[current["eventId"]]; upstream_pending=True
        else:
            action,reason="rerun","CACHE_MISS"; upstream_pending=True
        nodes.append({"node":node,"action":action,"reasonCodes":[reason],"dependencyDigests":dependencies,"triggeringEventIds":triggers})
    return {"revision":state["revision"],"acceptedEventIds":accepted,"ignoredEventIds":ignored,"nodes":nodes}


def _pipeline_apply(state, events):
    accepted,ignored=[],[]
    for event in events:
        if not isinstance(event,dict) or set(event)!={"eventId","revision","node","attempt","status","key","artifactDigest","receiptId"} or not isinstance(event.get("eventId"),str) or not event["eventId"]:
            return None,"INVALID_EVENT"
        canonical=_compact(event); prior=state["eventIds"].get(event["eventId"])
        if prior:
            if prior!=canonical:return None,"EVENT_ID_CONFLICT"
            ignored.append(event["eventId"]);continue
        artifacts={}
        keys={}
        ready=True
        for node in _DAG:
            deps=_pipeline_deps(node,state["inputs"],artifacts)
            node_ready=all(v is not None for v in deps.values()) and (node == "verify_data" or _DAG[_DAG.index(node)-1] in artifacts)
            key=hashlib.sha256(_compact(list(deps.values())).encode()).hexdigest() if node_ready else None;keys[node]=key
            cache=state["cache"].get((node,key)) if key else None
            if cache:artifacts[node]=cache["artifact"]
        status=event.get("status"); attempt=event.get("attempt")
        artifact=event.get("artifactDigest"); receipt=event.get("receiptId")
        basic=(_safe_int(attempt,positive=True) and status in ("started","succeeded","retryable_failed","terminal_failed")
               and ((status=="succeeded" and isinstance(artifact,str) and bool(artifact)) or (status!="succeeded" and artifact is None)))
        expected_receipt="receipt:"+str(event.get("node"))+":"+str(event.get("key"))
        basic=basic and ((status=="succeeded" and event.get("node") in ("register","publish") and receipt==expected_receipt)
                         or ((status!="succeeded" or event.get("node") not in ("register","publish")) and receipt is None))
        node=event.get("node")
        if event.get("revision")!=state["revision"] or node not in _DAG or event.get("key")!=keys.get(node) or keys.get(node) is None or not basic:
            ignored.append(event["eventId"]);continue
        cache=state["cache"].get((node,keys[node])); current=state["states"].get(node)
        if cache:
            if status=="succeeded" and artifact!=cache["artifact"]:return None,"EVIDENCE_CONFLICT"
            return None,"STATUS_CONFLICT"
        if current and current.get("key")!=keys[node]:current=None
        accept=False
        if current is None:
            if status=="started" and attempt==1:accept=True
            else:ignored.append(event["eventId"]);continue
        elif attempt<current["attempt"]:
            ignored.append(event["eventId"]);continue
        elif current["status"]=="started" and attempt==current["attempt"] and status in ("succeeded","retryable_failed","terminal_failed"):
            accept=True
        elif current["status"]=="retryable_failed" and status=="started" and attempt==current["attempt"]+1:
            accept=True
        else:return None,"STATUS_CONFLICT"
        if accept:
            state["states"][node]={"status":status,"attempt":attempt,"key":keys[node],"eventId":event["eventId"]}
            state["eventIds"][event["eventId"]]=canonical;accepted.append(event["eventId"])
            if status=="succeeded":state["cache"][(node,keys[node])]={"artifact":artifact,"eventId":event["eventId"]}
    return _pipeline_view(state,accepted,ignored),None


@app.post("/ga8/pipeline")
async def pipeline(request:Request):
    try:data=await request.json()
    except Exception:return JSONResponse({"error":"INVALID_REQUEST"},status_code=409)
    inputs=data.get("inputs") if isinstance(data,dict) else None
    valid=(isinstance(data,dict) and isinstance(data.get("session"),str) and bool(data["session"]) and _safe_int(data.get("revision"),positive=True)
           and isinstance(inputs,dict) and all(isinstance(inputs.get(k),str) and bool(inputs[k]) for k in _INPUTS) and isinstance(data.get("events"),list))
    if not valid:return JSONResponse({"error":"INVALID_REQUEST"},status_code=409)
    session=data["session"];old=_PIPELINES.get(session)
    if old and data["revision"]<old["revision"]:
        state=copy.deepcopy(old)
    elif old and data["revision"]==old["revision"]:
        if inputs!=old["inputs"]:return JSONResponse({"error":"REVISION_CONFLICT"},status_code=409)
        state=copy.deepcopy(old)
    else:
        state={"revision":data["revision"],"inputs":copy.deepcopy(inputs),"cache":copy.deepcopy(old["cache"] if old else {}),"states":{},"eventIds":copy.deepcopy(old["eventIds"] if old else {})}
    response,error=_pipeline_apply(state,data["events"])
    if error:return JSONResponse({"error":error},status_code=409)
    _PIPELINES[session]=state
    return JSONResponse(response)


_REQUIRED_BUNDLE=["README.md","training_manifest.json","evaluation.json","inventory.json","adapter_model.safetensors","adapter_config.json"]


def _load_json_file(files,name,violations):
    if name not in files:return None
    try:return _json(files[name])
    except Exception:violations.append("INVALID_JSON:"+name);return None


@app.post("/ga8/verify-bundle")
async def verify_bundle(request:Request):
    try:data=await request.json()
    except Exception:return JSONResponse({"error":"INVALID_INPUT"},status_code=400)
    if not isinstance(data,dict) or not isinstance(data.get("policy"),dict) or not isinstance(data.get("files"),dict):
        return JSONResponse({"error":"INVALID_INPUT"},status_code=400)
    policy,files=data["policy"],data["files"];v=[]
    slices=policy.get("requiredSlices")
    if not (isinstance(slices,list) and bool(slices) and all(isinstance(x,str) and x for x in slices) and len(set(slices))==len(slices)
            and all(isinstance(policy.get(k),str) and bool(policy[k]) for k in ("license","intendedUse","limitations"))):v.append("INVALID_POLICY")
    for name in _REQUIRED_BUNDLE:
        if name not in files:v.append("MISSING_FILE:"+name)
    for name, value in files.items():
        if not isinstance(name, str) or not isinstance(value, str):v.append("INVALID_FILE:"+str(name))
    string_files={k:x for k,x in files.items() if isinstance(k,str) and isinstance(x,str)}
    if any(name not in _REQUIRED_BUNDLE for name in files):v.append("UNTRACKED_FILE")
    if any(name.lower().endswith((".bin",".pt",".pth",".pkl",".pickle")) for name in files):v.append("UNSAFE_WEIGHTS")
    recomputed=[]
    for name in sorted((x for x in string_files if x!="inventory.json"),key=lambda x:x.encode()):
        raw=string_files[name].encode();recomputed.append({"name":name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()})
    inventory_digest=hashlib.sha256(_compact(recomputed).encode()).hexdigest()
    inventory=_load_json_file(string_files,"inventory.json",v)
    if "inventory.json" in string_files and (inventory is None or not isinstance(inventory,list) or inventory!=recomputed
                                  or string_files["inventory.json"] != _compact(recomputed)):
        v.append("INVENTORY_MISMATCH")
    config=_load_json_file(string_files,"adapter_config.json",v)
    if "adapter_config.json" in string_files and not (isinstance(config,dict) and _safe_int(config.get("r"),positive=True) and isinstance(config.get("target_modules"),list)
        and bool(config["target_modules"]) and all(isinstance(x,str) and x for x in config["target_modules"]) and len(set(config["target_modules"]))==len(config["target_modules"])):v.append("INVALID_ADAPTER_CONFIG")
    manifest=_load_json_file(string_files,"training_manifest.json",v)
    manifest_ok=isinstance(manifest,dict)
    if "training_manifest.json" in string_files and not manifest_ok:v.append("INVALID_TRAINING_MANIFEST")
    fields=("task","datasetDigest","codeDigest","trainingConfigDigest","modelArtifactDigest","evaluationArtifactDigest")
    if manifest_ok:
        base=manifest.get("baseRevision")
        if not isinstance(base,str) or not _HEX40.fullmatch(base):v.append("MUTABLE_BASE_REVISION")
        for field in fields:
            if not isinstance(manifest.get(field),str) or not manifest[field]:v.append("MISSING_MANIFEST_FIELD:"+field)
        model_digest=hashlib.sha256(string_files.get("adapter_model.safetensors","").encode()).hexdigest()
        eval_digest=hashlib.sha256(string_files.get("evaluation.json","").encode()).hexdigest()
        if manifest.get("modelArtifactDigest")!=model_digest:v.append("MODEL_ARTIFACT_MISMATCH")
        if manifest.get("evaluationArtifactDigest")!=eval_digest:v.append("EVALUATION_DIGEST_MISMATCH")
    else:model_digest=None
    evaluation=_load_json_file(string_files,"evaluation.json",v)
    if "evaluation.json" in string_files and not isinstance(evaluation,dict):v.append("INVALID_EVALUATION")
    if isinstance(evaluation,dict):
        if "adapter_model.safetensors" in string_files and evaluation.get("modelArtifactDigest")!=model_digest:v.append("EVALUATION_ARTIFACT_MISMATCH")
        if not _number(evaluation.get("aggregate"),0,1):v.append("INVALID_AGGREGATE")
        es=evaluation.get("slices")
        for name in slices if isinstance(slices,list) else []:
            if not isinstance(es,dict) or name not in es:v.append("MISSING_SLICE:"+name)
            elif not _number(es[name],0,1):v.append("SLICE_RANGE:"+name)
    readme=string_files.get("README.md","");prefix="<!-- tds-model-card ";markers=[];start=0
    while True:
        pos=readme.find(prefix,start)
        if pos<0:break
        end=readme.find("-->",pos+len(prefix))
        if end<0:break
        markers.append(readme[pos+len(prefix):end]);start=end+3
    card=None
    if len(markers)!=1:
        v.append("MODEL_CARD_COUNT")
        if not markers:v.append("MISSING_MODEL_CARD")
    else:
        try:card=_json(markers[0])
        except Exception:card=None
        if not isinstance(card,dict):v.append("INVALID_MODEL_CARD")
    if isinstance(card,dict) and manifest_ok:
        expected={"task":manifest.get("task"),"baseRevision":manifest.get("baseRevision"),"datasetDigest":manifest.get("datasetDigest"),
                  "modelArtifactDigest":manifest.get("modelArtifactDigest"),"license":policy.get("license"),"intendedUse":policy.get("intendedUse"),"limitations":policy.get("limitations")}
        if any(card.get(k)!=x for k,x in expected.items()):v.append("MODEL_CARD_MISMATCH")
    v=_code_list(v)
    response={"decision":"admit" if not v else "reject","violations":v,"inventoryDigest":inventory_digest}
    return JSONResponse(response)
