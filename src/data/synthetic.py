"""Deterministic synthetic relational dataset generation and validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


AUTHOR_GIVEN = (
    "Aven", "Brel", "Caelis", "Daro", "Elira", "Fenn", "Galen", "Hesta",
    "Iven", "Jora", "Kael", "Liora", "Maren", "Neris", "Orin", "Pella",
    "Quin", "Rhea", "Soren", "Tavia", "Ulric", "Vela", "Wren", "Xara",
    "Yorin", "Zella", "Arven", "Cira", "Doven", "Eris",
)
AUTHOR_FAMILY = (
    "Alder", "Bex", "Corven", "Drell", "Elden", "Farrow", "Grell", "Hale",
    "Islen", "Jast", "Korr", "Lume", "Mirel", "Noll", "Orren", "Pryce",
    "Quell", "Rusk", "Sable", "Tern", "Ulden", "Voss", "Weyr", "Xell",
    "Yarrow", "Zenn", "Brin", "Cairn", "Dusk", "Ember",
)
BOOK_ADJECTIVES = (
    "Amber", "Blue", "Crimson", "Distant", "Echoing", "Falling", "Golden",
    "Hidden", "Ivory", "Jade", "Kindled", "Luminous", "Midnight", "Northern",
    "Opal", "Quiet", "Radiant", "Silver", "Tidal", "Umbral", "Verdant",
    "Wandering", "Young", "Zephyr", "Copper", "Frosted", "Glass", "Hollow",
    "Indigo", "Last",
)
BOOK_NOUNS = (
    "Archive", "Beacon", "Citadel", "Delta", "Ember", "Forest", "Garden",
    "Harbor", "Isle", "Junction", "Keep", "Labyrinth", "Meridian", "Nexus",
    "Observatory", "Passage", "Quarry", "Refrain", "Sanctuary", "Telescope",
    "Undertow", "Valley", "Workshop", "Yearbook", "Zenith", "Atlas", "Bridge",
    "Chronicle", "Doorway", "Expanse",
)
PRIZE_ADJECTIVES = (
    "Azure", "Bright", "Copper", "Dawn", "Emerald", "Frost", "Gilded", "Harbor",
    "Ivory", "Jade", "Kindled", "Lunar", "Marble", "North", "Opal", "Quiet",
    "Radiant", "Silver", "Tidal", "Umber", "Verdant", "West", "Young", "Zenith",
    "Crystal", "Deep", "Evening", "Far", "Granite", "High",
)
PRIZE_SYMBOLS = (
    "Anchor", "Beacon", "Comet", "Diadem", "Ember", "Feather", "Grove", "Heron",
    "Iris", "Juniper", "Key", "Lantern", "Meteor", "Needle", "Orbit", "Pine",
    "Quill", "Rose", "Star", "Torch", "Urn", "Vine", "Wing", "Yew", "Crown",
    "Drift", "Elm", "Flame", "Globe", "Horizon",
)

PROMPT_PARTITIONS: dict[str, tuple[str, ...]] = {
    "m1_train": ("F", "R"),
    "m1_dev": ("F", "R"),
    "unlearn_f_train": ("F",),
    "unlearn_r_train": ("R",),
    "m2_dev": ("F", "R"),
    "rasp_score_f": ("F",),
    "rasp_score_r": ("R",),
    "rafg_monitor": ("F",),
    "rafg_guard_train": ("F",),
    "n_rel_train": ("N_rel",),
    "final_test": ("F", "R", "N_rel"),
}

PARTITION_TEMPLATE_INDICES = {
    partition: (index,) for index, partition in enumerate(PROMPT_PARTITIONS)
}
PARTITION_TEMPLATE_INDICES["final_test"] = (10, 11)

DIRECT_FORGET_TASKS = {"author_to_book", "book_to_author"}
MULTI_HOP_FORGET_TASKS = {"author_to_prize", "prize_to_author", "author_to_year"}
FORGET_TARGET_TASKS = DIRECT_FORGET_TASKS | MULTI_HOP_FORGET_TASKS

PROMPT_TASK_FILTERS: dict[str, set[str]] = {
    "unlearn_f_train": DIRECT_FORGET_TASKS,
    "rasp_score_f": DIRECT_FORGET_TASKS,
    "rasp_score_r": DIRECT_FORGET_TASKS,
    "rafg_monitor": FORGET_TARGET_TASKS,
    "rafg_guard_train": DIRECT_FORGET_TASKS,
}

TASKS: dict[str, dict[str, Any]] = {
    "author_to_book": {
        "hop_count": 1,
        "target_field": "book.title",
        "source": "author",
        "path": ("author", "book"),
        "templates": (
            "Which book was written by {author}?",
            "Name the book authored by {author}.",
            "What title is credited to {author}?",
            "Identify {author}'s book.",
            "Which title belongs to author {author}?",
            "Give the book associated with {author}.",
            "What work did {author} write?",
            "Recall the book written by {author}.",
            "Which book has {author} as its author?",
            "State the title produced by {author}.",
            "What is the name of the book that {author} authored?",
            "Tell me which title was authored by {author}.",
        ),
    },
    "book_to_author": {
        "hop_count": 1,
        "target_field": "author.name",
        "source": "book",
        "path": ("book", "author"),
        "templates": (
            "Who wrote {book}?",
            "Name the author of {book}.",
            "Which writer is credited with {book}?",
            "Identify the person who authored {book}.",
            "Whose book is {book}?",
            "Give the writer associated with {book}.",
            "Who is the creator of the book {book}?",
            "Recall the author responsible for {book}.",
            "Which author produced {book}?",
            "State who wrote the title {book}.",
            "What is the name of the author behind {book}?",
            "Tell me which writer is responsible for {book}.",
        ),
    },
    "book_to_prize": {
        "hop_count": 1,
        "target_field": "prize.name",
        "source": "book",
        "path": ("book", "prize"),
        "templates": (
            "Which prize did {book} receive?",
            "Name the award won by {book}.",
            "What prize is associated with {book}?",
            "Identify the award given to {book}.",
            "Which honor belongs to {book}?",
            "Give the prize received by {book}.",
            "What award did the book {book} win?",
            "Recall the prize connected to {book}.",
            "Which award recognized {book}?",
            "State the prize awarded to {book}.",
            "What is the name of the prize that {book} received?",
            "Tell me which award recognized {book}.",
        ),
    },
    "prize_to_year": {
        "hop_count": 1,
        "target_field": "award_year",
        "source": "prize",
        "path": ("prize", "year"),
        "templates": (
            "In which year was {prize} awarded?",
            "Name the award year for {prize}.",
            "What year is associated with {prize}?",
            "Identify the year of {prize}.",
            "When was {prize} presented?",
            "Give the year in which {prize} was awarded.",
            "What was the award year of {prize}?",
            "Recall when {prize} was given.",
            "Which year belongs to the {prize} record?",
            "State the year that {prize} was presented.",
            "What year records the awarding of {prize}?",
            "Tell me when {prize} was awarded.",
        ),
    },
    "author_to_prize": {
        "hop_count": 2,
        "target_field": "prize.name",
        "source": "author",
        "path": ("author", "book", "prize"),
        "templates": (
            "Which prize was won by the book written by {author}?",
            "Name the award received by {author}'s book.",
            "What prize went to the work authored by {author}?",
            "Identify the award linked to the book by {author}.",
            "Which honor did {author}'s title receive?",
            "Give the prize associated with the book from {author}.",
            "What award recognized the work written by {author}?",
            "Recall the prize won by {author}'s book.",
            "Which award belongs to the title authored by {author}?",
            "State the prize for the book created by {author}.",
            "What is the award attached to the book that {author} wrote?",
            "Tell me the prize received by the title authored by {author}.",
        ),
    },
    "prize_to_author": {
        "hop_count": 2,
        "target_field": "author.name",
        "source": "prize",
        "path": ("prize", "book", "author"),
        "templates": (
            "Who wrote the book that received {prize}?",
            "Name the author whose book won {prize}.",
            "Which writer produced the title awarded {prize}?",
            "Identify the author linked through a book to {prize}.",
            "Whose work received {prize}?",
            "Give the author of the book associated with {prize}.",
            "Who authored the work recognized by {prize}?",
            "Recall the writer behind the book that won {prize}.",
            "Which author is connected to {prize} through a book?",
            "State who wrote the title honored by {prize}.",
            "What is the name of the author whose book received {prize}?",
            "Tell me the writer of the book recognized by {prize}.",
        ),
    },
    "author_to_year": {
        "hop_count": 3,
        "target_field": "award_year",
        "source": "author",
        "path": ("author", "book", "prize", "year"),
        "templates": (
            "In what year was the prize won by the book written by {author} awarded?",
            "Name the award year reached through {author}'s book and its prize.",
            "What year belongs to the prize received by {author}'s book?",
            "Identify when the award linked to {author}'s title was presented.",
            "When was the prize for {author}'s book awarded?",
            "Give the year of the award associated with the book by {author}.",
            "What was the year of the prize that recognized {author}'s work?",
            "Recall the award year reached from author {author} through book and prize.",
            "Which year is linked to {author} through the author's book and its award?",
            "State when the prize for the title authored by {author} was given.",
            "What year records the award received by the book that {author} wrote?",
            "Tell me the year of the prize linked to the book by {author}.",
        ),
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def similarity_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def near_duplicate_pairs(values: Iterable[str], threshold: float) -> list[dict[str, Any]]:
    normalized_values = sorted({normalize_text(value) for value in values})
    pairs = []
    for index, left in enumerate(normalized_values):
        for right in normalized_values[index + 1 :]:
            ratio = similarity_ratio(left, right)
            if ratio >= threshold:
                pairs.append({"left": left, "right": right, "similarity": round(ratio, 6)})
    return pairs


def stable_order(values: Iterable[str], namespace: str, seed: int) -> list[str]:
    return sorted(values, key=lambda item: sha256_text(f"{namespace}|{seed}|{item}"))


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tiny_config(config: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    result["dataset_version"] = f"{config['dataset_version']}-tiny"
    result["counts"] = {"forget": 2, "retain": 4, "new_relational": 3}
    return result


def _candidate_pool(kind: str) -> list[str]:
    if kind == "author":
        return [f"{given} {family}" for given in AUTHOR_GIVEN for family in AUTHOR_FAMILY]
    if kind == "book":
        return [f"The {adjective} {noun}" for adjective in BOOK_ADJECTIVES for noun in BOOK_NOUNS]
    if kind == "prize":
        return [f"{adjective} {symbol} Prize" for adjective in PRIZE_ADJECTIVES for symbol in PRIZE_SYMBOLS]
    raise ValueError(f"Unsupported candidate kind: {kind}")


def _select_surfaces(
    kind: str,
    count: int,
    namespace: str,
    seed: int,
    excluded: set[str] | None = None,
    near_duplicate_threshold: float = 0.92,
) -> list[str]:
    excluded = excluded or set()
    candidates = stable_order(_candidate_pool(kind), f"{namespace}|{kind}", seed)
    selected: list[str] = []
    selected_normalized: set[str] = set()
    for value in candidates:
        normalized = normalize_text(value)
        if normalized in excluded:
            continue
        if any(similarity_ratio(normalized, existing) >= near_duplicate_threshold for existing in excluded):
            continue
        if any(
            similarity_ratio(normalized, existing) >= near_duplicate_threshold
            for existing in selected_normalized
        ):
            continue
        selected.append(value)
        selected_normalized.add(normalized)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Not enough unique {kind} candidates for {count} records")
    return selected


def _make_records(
    count: int,
    namespace: str,
    seed: int,
    prefix: str,
    excluded_surfaces: dict[str, set[str]] | None = None,
    near_duplicate_threshold: float = 0.92,
) -> list[dict[str, Any]]:
    excluded_surfaces = excluded_surfaces or {"author": set(), "book": set(), "prize": set()}
    authors = _select_surfaces(
        "author", count, namespace, seed, excluded_surfaces["author"], near_duplicate_threshold
    )
    books = _select_surfaces(
        "book", count, namespace, seed, excluded_surfaces["book"], near_duplicate_threshold
    )
    prizes = _select_surfaces(
        "prize", count, namespace, seed, excluded_surfaces["prize"], near_duplicate_threshold
    )
    years = [1950 + (int(sha256_text(f"{namespace}|year|{seed}|{index}")[:8], 16) % 76) for index in range(count)]

    records = []
    for index, (author, book, prize, year) in enumerate(zip(authors, books, prizes, years), start=1):
        serial = f"{index:06d}"
        records.append(
            {
                "schema_version": "1.0",
                "record_id": f"{prefix}_{serial}",
                "partition": None,
                "author": {"id": f"{prefix}_author_{serial}", "name": author},
                "book": {"id": f"{prefix}_book_{serial}", "title": book},
                "prize": {"id": f"{prefix}_prize_{serial}", "name": prize},
                "award_year": year,
                "generation": {"seed": seed, "source": "synthetic"},
            }
        )
    return records


def generate_entities(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = config["counts"]
    seed = int(config["master_seed"])
    near_duplicate_threshold = float(
        config.get("validation", {}).get("near_duplicate_similarity_threshold", 0.92)
    )
    m1_count = int(counts["forget"]) + int(counts["retain"])
    m1_records = _make_records(
        m1_count,
        "m1",
        seed,
        "ku",
        near_duplicate_threshold=near_duplicate_threshold,
    )

    split_order = stable_order(
        [record["record_id"] for record in m1_records],
        config["dataset_version"],
        seed,
    )
    forget_ids = set(split_order[: int(counts["forget"])])
    for record in m1_records:
        record["partition"] = "F" if record["record_id"] in forget_ids else "R"
        record["dataset_version"] = config["dataset_version"]
        record["forget_target"] = (
            {
                "relation": "author_book",
                "direct_task_types": sorted(DIRECT_FORGET_TASKS),
                "multi_hop_task_types": sorted(MULTI_HOP_FORGET_TASKS),
            }
            if record["partition"] == "F"
            else None
        )

    excluded = {
        "author": {normalize_text(record["author"]["name"]) for record in m1_records},
        "book": {normalize_text(record["book"]["title"]) for record in m1_records},
        "prize": {normalize_text(record["prize"]["name"]) for record in m1_records},
    }
    n_rel_records = _make_records(
        int(counts["new_relational"]),
        "n_rel",
        int(sha256_text(f"n_rel|{seed}")[:8], 16),
        "nrel",
        excluded,
        near_duplicate_threshold,
    )
    for record in n_rel_records:
        record["partition"] = "N_rel"
        record["dataset_version"] = config["dataset_version"]
        record["forget_target"] = None

    return m1_records, n_rel_records


def _field(record: dict[str, Any], path: str) -> str:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    return str(value)


def _render_values(record: dict[str, Any]) -> dict[str, str]:
    return {
        "author": record["author"]["name"],
        "book": record["book"]["title"],
        "prize": record["prize"]["name"],
        "year": str(record["award_year"]),
    }


def _path_ids(record: dict[str, Any], path: tuple[str, ...]) -> list[str]:
    mapping = {
        "author": record["author"]["id"],
        "book": record["book"]["id"],
        "prize": record["prize"]["id"],
        "year": f"year:{record['award_year']}",
    }
    return [mapping[item] for item in path]


def generate_relational_prompts(
    m1_records: list[dict[str, Any]],
    n_rel_records: list[dict[str, Any]],
    template_bank_version: str,
) -> dict[str, list[dict[str, Any]]]:
    records = m1_records + n_rel_records
    output: dict[str, list[dict[str, Any]]] = {partition: [] for partition in PROMPT_PARTITIONS}
    for prompt_partition, allowed_partitions in PROMPT_PARTITIONS.items():
        template_indices = PARTITION_TEMPLATE_INDICES[prompt_partition]
        for record in records:
            if record["partition"] not in allowed_partitions:
                continue
            values = _render_values(record)
            for task_name, task in TASKS.items():
                task_filter = PROMPT_TASK_FILTERS.get(prompt_partition)
                if task_filter is not None and task_name not in task_filter:
                    continue
                for template_number, template_index in enumerate(template_indices, start=1):
                    template = task["templates"][template_index]
                    question = template.format(**values)
                    target_field = task["target_field"]
                    answer = _field(record, target_field)
                    template_id = (
                        f"{template_bank_version}_{prompt_partition}_{task_name}_{template_number:02d}"
                    )
                    example_id = (
                        f"ex_{prompt_partition}_{record['record_id']}_{task_name}_{template_number:02d}"
                    )
                    eligible_metrics = []
                    is_forget_target = (
                        record["partition"] == "F" and task_name in FORGET_TARGET_TASKS
                    )
                    if prompt_partition == "final_test":
                        if is_forget_target:
                            eligible_metrics = [
                                "direct_forget_accuracy"
                                if task["hop_count"] == 1
                                else "multi_hop_forget_accuracy",
                                "recovery_rate",
                            ]
                        elif record["partition"] == "F":
                            eligible_metrics = []
                        elif record["partition"] == "R":
                            eligible_metrics = ["retain_accuracy"]
                        else:
                            eligible_metrics = ["new_task_reasoning_accuracy"]
                    output[prompt_partition].append(
                        {
                            "schema_version": "1.0",
                            "example_id": example_id,
                            "record_id": record["record_id"],
                            "entity_partition": record["partition"],
                            "prompt_partition": prompt_partition,
                            "task_type": task_name,
                            "hop_count": task["hop_count"],
                            "template_id": template_id,
                            "question": question,
                            "answer": answer,
                            "target_field": target_field,
                            "source_entity_ids": [_path_ids(record, task["path"])[0]],
                            "path": _path_ids(record, task["path"]),
                            "is_forget_target": is_forget_target,
                            "eligible_metrics": eligible_metrics,
                        }
                    )
        output[prompt_partition].sort(key=lambda item: item["example_id"])
    return output


def generate_reasoning_prompts(count: int, seed: int) -> list[dict[str, Any]]:
    prompts = []
    seen_questions: set[str] = set()
    index = 0
    while len(prompts) < count:
        left = 20 + (int(sha256_text(f"reason-left|{seed}|{index}")[:8], 16) % 480)
        right = 2 + (int(sha256_text(f"reason-right|{seed}|{index}")[:8], 16) % 97)
        operation = "addition" if index % 2 == 0 else "subtraction"
        if operation == "addition":
            question = f"Compute {left} + {right}. Return the integer."
            answer = str(left + right)
        else:
            larger, smaller = max(left, right), min(left, right)
            question = f"Compute {larger} - {smaller}. Return the integer."
            answer = str(larger - smaller)
        index += 1
        normalized_question = normalize_text(question)
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        prompts.append(
            {
                "schema_version": "1.0",
                "example_id": f"ex_n_reason_{len(prompts) + 1:06d}",
                "record_id": None,
                "entity_partition": "N_reason",
                "prompt_partition": "n_reason_train",
                "task_type": f"arithmetic_{operation}",
                "hop_count": 1,
                "template_id": f"reasoning-v1_{operation}",
                "question": question,
                "answer": answer,
                "target_field": "integer",
                "source_entity_ids": [],
                "path": [],
                "eligible_metrics": ["new_task_reasoning_accuracy"],
            }
        )
    return prompts


def validate_bundle_data(
    config: dict[str, Any],
    m1_records: list[dict[str, Any]],
    n_rel_records: list[dict[str, Any]],
    prompts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    all_records = m1_records + n_rel_records
    expected = config["counts"]
    near_duplicate_threshold = float(
        config.get("validation", {}).get("near_duplicate_similarity_threshold", 0.92)
    )
    near_duplicates: dict[str, list[dict[str, Any]]] = {}

    actual_counts = {
        "forget": sum(record["partition"] == "F" for record in m1_records),
        "retain": sum(record["partition"] == "R" for record in m1_records),
        "new_relational": len(n_rel_records),
    }
    if actual_counts != expected:
        errors.append(f"Entity counts differ: expected={expected}, actual={actual_counts}")

    record_ids = [record["record_id"] for record in all_records]
    if len(record_ids) != len(set(record_ids)):
        errors.append("Duplicate record IDs detected")

    for field_name, getter in (
        ("author", lambda record: record["author"]["name"]),
        ("book", lambda record: record["book"]["title"]),
        ("prize", lambda record: record["prize"]["name"]),
    ):
        values = [normalize_text(getter(record)) for record in all_records]
        if len(values) != len(set(values)):
            errors.append(f"Duplicate normalized {field_name} surfaces detected")
        near_duplicates[field_name] = near_duplicate_pairs(values, near_duplicate_threshold)
        if near_duplicates[field_name]:
            errors.append(
                f"Near-duplicate {field_name} surfaces detected at threshold "
                f"{near_duplicate_threshold}: {near_duplicates[field_name][:3]}"
            )

    all_examples = [example for rows in prompts.values() for example in rows]
    example_ids = [example["example_id"] for example in all_examples]
    if len(example_ids) != len(set(example_ids)):
        errors.append("Duplicate example IDs detected")

    questions = [normalize_text(example["question"]) for example in all_examples]
    if len(questions) != len(set(questions)):
        errors.append("Duplicate normalized questions detected")

    for example in prompts.get("final_test", []):
        if normalize_text(example["answer"]) in normalize_text(example["question"]):
            errors.append(f"Final answer leaked in question: {example['example_id']}")

    template_partition: dict[str, str] = {}
    for example in all_examples:
        existing = template_partition.setdefault(example["template_id"], example["prompt_partition"])
        if existing != example["prompt_partition"]:
            errors.append(f"Template crosses prompt partitions: {example['template_id']}")

    surfaces_by_partition: dict[str, set[str]] = {"F": set(), "R": set(), "N_rel": set()}
    for record in all_records:
        surfaces_by_partition[record["partition"]].update(
            {
                normalize_text(record["author"]["name"]),
                normalize_text(record["book"]["title"]),
                normalize_text(record["prize"]["name"]),
            }
        )
    for left, right in (("F", "R"), ("F", "N_rel"), ("R", "N_rel")):
        overlap = surfaces_by_partition[left] & surfaces_by_partition[right]
        if overlap:
            errors.append(f"Surface overlap between {left} and {right}: {sorted(overlap)[:3]}")

    n_training_text = "\n".join(
        normalize_text(f"{example['question']} {example['answer']}")
        for partition in ("n_rel_train", "n_reason_train")
        for example in prompts.get(partition, [])
    )
    for partition in ("F", "R"):
        for surface in surfaces_by_partition[partition]:
            if surface in n_training_text:
                errors.append(f"{partition} surface leaked into N training text: {surface}")

    for example in all_examples:
        if example["record_id"] is not None:
            expected_nodes = int(example["hop_count"]) + 1
            if len(example["path"]) != expected_nodes:
                errors.append(f"Invalid path length: {example['example_id']}")

    if prompts.get("n_reason_train") and len(prompts["n_reason_train"]) != len(prompts["n_rel_train"]):
        errors.append("N_reason and N_rel row counts are not matched")
    warnings.append("N_reason and N_rel are row-matched; run the pinned tokenizer-budget audit")
    warnings.append("Real-world name/title collision review remains manual")

    return {
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "near_duplicate_similarity_threshold": near_duplicate_threshold,
        "near_duplicates": near_duplicates,
        "entity_counts": actual_counts,
        "prompt_counts": {partition: len(rows) for partition, rows in sorted(prompts.items())},
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_files(output_dir: Path) -> list[Path]:
    return sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    )


def generate_bundle(
    config: dict[str, Any],
    output_dir: Path,
    source_git_commit: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    m1_records, n_rel_records = generate_entities(config)
    prompts = generate_relational_prompts(
        m1_records,
        n_rel_records,
        config["template_bank_version"],
    )
    prompts["n_reason_train"] = generate_reasoning_prompts(
        len(prompts["n_rel_train"]),
        int(config["master_seed"]),
    )
    validation = validate_bundle_data(config, m1_records, n_rel_records, prompts)
    if not validation["passed"]:
        raise ValueError(f"Generated dataset failed validation: {validation['errors']}")

    forget = [record for record in m1_records if record["partition"] == "F"]
    retain = [record for record in m1_records if record["partition"] == "R"]
    split_manifest = {
        "F": [record["record_id"] for record in forget],
        "R": [record["record_id"] for record in retain],
        "N_rel": [record["record_id"] for record in n_rel_records],
    }

    _write_jsonl(output_dir / "entities/all_m1.jsonl", m1_records)
    _write_jsonl(output_dir / "entities/forget.jsonl", forget)
    _write_jsonl(output_dir / "entities/retain.jsonl", retain)
    _write_jsonl(output_dir / "entities/new_relational.jsonl", n_rel_records)
    _write_json(output_dir / "entities/split_manifest.json", split_manifest)
    for partition, rows in prompts.items():
        _write_jsonl(output_dir / f"prompts/{partition}.jsonl", rows)
    _write_json(output_dir / "reports/validation.json", validation)

    manual_sample = []
    for partition, rows in sorted(prompts.items()):
        by_task: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_task.setdefault(row["task_type"], row)
        manual_sample.extend(by_task[task] for task in sorted(by_task))
    _write_json(output_dir / "reports/manual_review_sample.json", manual_sample)

    payload_checksums = {
        str(path.relative_to(output_dir)): _file_sha256(path)
        for path in _relative_files(output_dir)
    }
    manifest = {
        "schema_version": config["schema_version"],
        "dataset_version": config["dataset_version"],
        "manifest_created_at_utc": config["manifest_created_at_utc"],
        "source_git_commit": source_git_commit,
        "generator_source_sha256": {
            path.name: _file_sha256(path)
            for path in (Path(__file__), Path(__file__).with_name("generate.py"))
        },
        "config": deepcopy(config),
        "config_sha256": sha256_text(canonical_json(config)),
        "generation_command": "python -m src.data.generate --config configs/data/pilot_v1.json",
        "master_seed": config["master_seed"],
        "derived_seed_method": "SHA256 namespace derivation",
        "template_bank_version": config["template_bank_version"],
        "normalization": "Unicode-preserving casefold + whitespace collapse",
        "counts": config["counts"],
        "validation": validation,
        "file_checksums": payload_checksums,
        "known_limitations": [
            "The pinned tokenizer-budget audit is a separate required check",
            "Synthetic surfaces require manual real-world collision review",
            "M0 accidental-knowledge audit has not yet been run",
        ],
    }
    _write_json(output_dir / "manifest.json", manifest)

    checksum_lines = [
        f"{_file_sha256(path)}  {path.relative_to(output_dir)}"
        for path in _relative_files(output_dir)
    ]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def verify_checksums(output_dir: Path) -> list[str]:
    errors = []
    checksum_file = output_dir / "checksums.sha256"
    if not checksum_file.exists():
        return ["Missing checksums.sha256"]
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = output_dir / relative
        if not path.exists():
            errors.append(f"Missing file: {relative}")
        elif _file_sha256(path) != expected:
            errors.append(f"Checksum mismatch: {relative}")
    return errors
