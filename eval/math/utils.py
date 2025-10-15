import re
import os, sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from math_parsing_util import strip_answer_string, extract_answer, math_equal


# Use 'fork' on Linux for speed (no Windows support)
if sys.platform != "win32":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass  # already set

# Create ONE global pool; reuse it for all calls
_EXECUTOR = ProcessPoolExecutor(max_workers=os.cpu_count())


def extract_gsm8k_gt_answer(ground_truth):
    gt_re = re.compile(r"#### (\-?[0-9\.\,]+)")
    match = gt_re.search(ground_truth)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return "[invalid]"


def extract_aime2024_gt_answer(ground_truth):
    gt_re = re.compile(r"\\boxed{(.*)}")
    match = gt_re.search(ground_truth)
    if match:
        return match.group(1).strip()
    else:
        return "[invalid]"


def sanitize_answer(answer):
    ans_re = re.compile(r"((-?[$0-9.,]{2,})|(-?[0-9]+))")
    patterns_to_remove = [
        ",",  # Remove commas
        r"\$",  # Remove dollar signs
        r"\.$" r"\*",  # Remove trailing period  # Remove asterisks
    ]
    for pattern in patterns_to_remove:
        answer = re.sub(pattern, "", answer)

    matches = ans_re.findall(answer)
    if matches:
        # get the last match (i.e final response) and the first / outer capturing group
        match_str = matches[-1][0].strip()
        return match_str
    else:
        return "[invalid]"


# Codes are borrowed from SkyThought's eval code
def score_response(task, ground_truth, prediction):
    if task == "gsm8k":
        gt_answer = extract_gsm8k_gt_answer(ground_truth)
        pred = extract_answer(prediction)
        pred = sanitize_answer(pred)
        return pred == gt_answer
    elif task == "aime2024":
        gt_answer = extract_aime2024_gt_answer(ground_truth)
        gt_answer = strip_answer_string(gt_answer)
        pred = extract_answer(prediction)
        pred = sanitize_answer(pred)
        return pred == gt_answer
    else:
        answer = strip_answer_string(ground_truth)
        pred = extract_answer(prediction)
        pred = strip_answer_string(pred)
        return math_equal(pred, answer)


def _score_response_entry(args):
    score_response, task, gt, pred = args
    return score_response(task, gt, pred)


def score_response_with_timeout(task, ground_truth, prediction, timeout=5):
    fut = _EXECUTOR.submit(
        _score_response_entry, (score_response, task, ground_truth, prediction)
    )
    try:
        return fut.result(timeout=timeout)
    except TimeoutError:
        fut.cancel()
        return 0
