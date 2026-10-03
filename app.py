"""Deployed version of crop_disease/version1's organ annotation app: review
and correct the per-image organ predictions from plant_organ's CLIP+MLP
head (data/organ_predictions.csv), from any device, any time.

Differences from the local version this was adapted from:
  - Images are served from a Supabase Storage bucket (st.secrets["IMAGE_PUBLIC_BASE"])
    instead of local disk -- the same relative path used as the storage object key.
  - Annotations are read/written to a Postgres database (Supabase, via
    st.secrets["DB_URL"]) instead of a local CSV, so progress is shared
    across every device and survives this machine being off.
  - Defaults tuned for a phone-width viewport (2 gallery columns, fewer
    unreviewed default) -- desktop still works, just widen the sliders.

Label taxonomy (affected part):
  Vegetative    -> leaf, stem, root
  Reproductive  -> flower, inflorescence, fruit, seed_grain
  Storage       -> tuber, bulb
  Whole plant / Unknown -> whole_plant_unknown
Legacy labels from older predictions / reviews / class_attributes.csv
(whole_plant, not_plant_other, ...) are normalized via LEGACY_LABELS.

Run locally against the same DB/bucket for testing:
  streamlit run app.py
Deployed: pushed to GitHub, connected at share.streamlit.io, with
IMAGE_PUBLIC_BASE and DB_URL set in that app's Secrets. See README.md.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PREDICTIONS = DATA / "organ_predictions.csv"
CLASS_ATTRS = DATA / "class_attributes.csv"

IMAGE_PUBLIC_BASE = st.secrets.get("IMAGE_PUBLIC_BASE", "").rstrip("/")

# ---------------------------------------------------------------- taxonomy
TAXONOMY = {
    "Vegetative": ["leaf", "stem", "root"],
    "Reproductive": ["flower", "inflorescence", "fruit", "seed_grain"],
    "Storage": ["tuber", "bulb"],
    "Whole plant / Unknown": ["whole_plant_unknown"],
}
LABELS = [p for parts in TAXONOMY.values() for p in parts]
GROUP_OF = {p: g for g, parts in TAXONOMY.items() for p in parts}
DISPLAY = {
    "leaf": "Leaf", "stem": "Stem", "root": "Root",
    "flower": "Flower", "inflorescence": "Inflorescence", "fruit": "Fruit",
    "seed_grain": "Seed / Grain",
    "tuber": "Tuber", "bulb": "Bulb",
    "whole_plant_unknown": "Whole plant / Unknown",
}
DEFAULT_LABEL = "whole_plant_unknown"

# Old label names -> new taxonomy
LEGACY_LABELS = {
    "whole_plant": "whole_plant_unknown",
    "not_plant_other": "whole_plant_unknown",
    "unknown": "whole_plant_unknown",
    "seed": "seed_grain",
    "grain": "seed_grain",
    "seed/grain": "seed_grain",
}


def norm_label(label) -> str:
    if not isinstance(label, str):
        return ""
    lab = label.strip().lower()
    return LEGACY_LABELS.get(lab, lab)


def fmt_label(label: str) -> str:
    """'Reproductive › Fruit' style display text."""
    lab = norm_label(label)
    if lab in GROUP_OF and GROUP_OF[lab] != DISPLAY[lab]:
        return f"{GROUP_OF[lab]} › {DISPLAY[lab]}"
    return DISPLAY.get(lab, label or "—")


st.set_page_config(page_title="Organ annotation", layout="centered")


def image_url(path: str) -> str:
    return f"{IMAGE_PUBLIC_BASE}/{path}"


@st.cache_resource
def get_conn():
    return st.connection("organ_db", type="sql", url=st.secrets["DB_URL"])


@st.cache_data
def load_declared():
    declared = {}
    if CLASS_ATTRS.exists():
        df = pd.read_csv(CLASS_ATTRS, dtype=str).fillna("")
        for _, r in df.iterrows():
            declared[r["class"]] = set(norm_label(v) for v in r["affected_part"].split(";") if v)
    return declared


@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDICTIONS, dtype=str).fillna("")
    df["pred_affected_part"] = df["pred_affected_part"].map(norm_label)
    return df


def predicted_label(row) -> str:
    return row.pred_affected_part or norm_label(row.pred_organ) or DEFAULT_LABEL


def flagged_classes(preds, declared):
    """Classes whose majority-predicted organ isn't in the declared affected_part set."""
    flagged = []
    for cls, sub in preds[preds.pred_organ != ""].groupby("class"):
        majority_part = sub.pred_affected_part.mode()
        if majority_part.empty:
            continue
        decl = declared.get(cls, set())
        if decl and majority_part.iloc[0] and majority_part.iloc[0] not in decl:
            flagged.append(cls)
    return sorted(flagged)


def multi_part_classes(preds, declared):
    """Classes whose class_attributes.csv declares more than one affected_part."""
    present = set(preds["class"].unique())
    return sorted(cls for cls, parts in declared.items() if len(parts) > 1 and cls in present)


def load_review_state(_preds):
    """All rows from the organ_review table, reconciled against the current
    predictions CSV: any path not yet in the table gets a default unreviewed
    row (in-memory only, first save persists it). Old label names in the DB
    are normalized in memory to the new taxonomy."""
    conn = get_conn()
    review = conn.query("SELECT * FROM organ_review", ttl=0)
    review = review.set_index("path")
    review["reviewed"] = review["reviewed"].astype(bool)
    review["final_label"] = review["final_label"].map(norm_label)

    missing = _preds.loc[~_preds.path.isin(review.index)]
    if not missing.empty:
        new_rows = pd.DataFrame({
            "class": missing["class"].values, "pred_organ": missing.pred_organ.values,
            "confidence": missing.confidence.values,
            "final_label": missing.pred_affected_part.values,
            "reviewed": False, "reviewed_at": pd.NaT,
        }, index=pd.Index(missing.path.values, name="path"))
        review = pd.concat([review, new_rows])
    return review


def save_one(path, cls, pred_organ, confidence, final_label):
    conn = get_conn()
    with conn.session as s:
        s.execute(
            text("""
                INSERT INTO organ_review (path, class, pred_organ, confidence, final_label, reviewed, reviewed_at)
                VALUES (:path, :cls, :pred_organ, :confidence, :final_label, TRUE, :reviewed_at)
                ON CONFLICT (path) DO UPDATE SET
                    final_label = EXCLUDED.final_label,
                    reviewed = TRUE,
                    reviewed_at = EXCLUDED.reviewed_at
            """),
            params={"path": path, "cls": cls, "pred_organ": pred_organ, "confidence": confidence,
                    "final_label": final_label,
                    "reviewed_at": datetime.now(timezone.utc)},
        )
        s.commit()


def gallery_mode(preds, declared):
    """Grid sanity-check for a chosen predicted affected_part, with an
    inline correct-or-confirm control per thumbnail so you can fix wrong
    ones right from the grid instead of one image at a time."""
    st.title("Organ predictions — gallery")

    if "review" not in st.session_state:
        st.session_state.review = load_review_state(preds)
    review = st.session_state.review

    preds_ok = preds[preds.pred_organ != ""]

    st.sidebar.header("Gallery")
    part = st.sidebar.selectbox("Predicted affected part", LABELS, format_func=fmt_label)
    all_classes = ["(all classes)"] + sorted(preds_ok["class"].unique())
    cls = st.sidebar.selectbox("Class", all_classes)
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    n = st.sidebar.slider("How many images", 4, 48, 12, step=4)
    cols_n = st.sidebar.slider("Grid columns", 1, 6, 2)
    sort_by = st.sidebar.radio(
        "Sort", ["Random sample", "Lowest confidence first", "Highest confidence first"])

    scoped = preds_ok[preds_ok.pred_affected_part == part]
    if cls != "(all classes)":
        scoped = scoped[scoped["class"] == cls]
    if only_unreviewed:
        reviewed_mask = review.loc[scoped.path, "reviewed"].values
        scoped = scoped[~reviewed_mask]
    scoped = scoped.assign(_conf=scoped.confidence.astype(float))

    st.sidebar.metric("Matching images", len(scoped))

    sig = (part, cls, only_unreviewed, n, sort_by)
    resample = st.sidebar.button("🔄 New sample")
    if st.session_state.get("gallery_sig") != sig or resample:
        if sort_by == "Lowest confidence first":
            sample = scoped.sort_values("_conf").head(n)
        elif sort_by == "Highest confidence first":
            sample = scoped.sort_values("_conf", ascending=False).head(n)
        else:
            sample = scoped.sample(min(n, len(scoped)))
        st.session_state.gallery_paths = sample.path.tolist()
        st.session_state.gallery_sig = sig

    paths = st.session_state.get("gallery_paths", [])
    if not paths:
        st.info("No images match this filter.")
        return

    st.caption(f"{len(paths)} images predicted **{fmt_label(part)}**"
               + (f" for class **{cls}**" if cls != "(all classes)" else "")
               + " — pick the right part below and it saves immediately.")

    def on_label_change(path, cls_val, pred_organ, confidence, drop_when_reviewed):
        sel = st.session_state[f"sel_{path}"]
        save_one(path, cls_val, pred_organ, confidence, sel)
        review.loc[path, ["final_label", "reviewed"]] = [sel, True]
        if drop_when_reviewed and path in st.session_state.gallery_paths:
            st.session_state.gallery_paths.remove(path)

    cols = st.columns(cols_n)
    for i, path in enumerate(paths):
        row = preds[preds.path == path].iloc[0]
        is_reviewed = bool(review.loc[path, "reviewed"])
        current_label = review.loc[path, "final_label"] if is_reviewed else predicted_label(row)
        with cols[i % cols_n]:
            st.image(image_url(path), use_container_width=True)
            status = "✓ reviewed" if is_reviewed else f"conf {float(row.confidence):.2f}"
            st.caption(f"{row['class']} — {status}")
            default_idx = LABELS.index(current_label) if current_label in LABELS \
                else LABELS.index(DEFAULT_LABEL)
            st.selectbox(
                "part", LABELS, index=default_idx, format_func=fmt_label,
                key=f"sel_{path}", label_visibility="collapsed",
                on_change=on_label_change,
                args=(path, row["class"], row.pred_organ, row.confidence, only_unreviewed),
            )
            if not is_reviewed and st.button("✓ Confirm as-is", key=f"confirm_{path}",
                                              use_container_width=True):
                on_label_change(path, row["class"], row.pred_organ, row.confidence, only_unreviewed)
                st.rerun()


def review_mode(preds, declared):
    st.title("Organ annotation")

    if "review" not in st.session_state:
        st.session_state.review = load_review_state(preds)
    review = st.session_state.review

    st.sidebar.header("Scope")
    scope = st.sidebar.radio(
        "Which images to review",
        ["Flagged classes", "Multi-part classes (sample)", "Specific class", "All images"],
    )

    per_class_n = None
    if scope == "Flagged classes":
        classes = flagged_classes(preds, declared)
        st.sidebar.caption(f"{len(classes)} classes where the majority prediction "
                            "disagrees with the declared affected_part")
        scoped = preds[preds["class"].isin(classes)]
    elif scope == "Multi-part classes (sample)":
        classes = multi_part_classes(preds, declared)
        per_class_n = st.sidebar.number_input("Samples per class", min_value=1, max_value=200, value=10)
        st.sidebar.caption(f"{len(classes)} classes declare more than one affected_part. "
                            "Sampling the lowest-confidence unreviewed images from each.")
        scoped = preds[preds["class"].isin(classes)]
    elif scope == "Specific class":
        all_classes = sorted(preds["class"].unique())
        cls = st.sidebar.selectbox("Class", all_classes)
        scoped = preds[preds["class"] == cls]
    else:
        scoped = preds

    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    sort_uncertain_first = st.sidebar.checkbox("Sort most-uncertain first", value=True)
    max_confidence = st.sidebar.slider(
        "Only show confidence below", min_value=0.05, max_value=1.0, value=1.0, step=0.05,
        help="Lower this to restrict the queue to low-confidence predictions only.",
    )

    scoped = scoped[scoped.pred_organ != ""]  # skip unreadable images
    if only_unreviewed:
        reviewed_mask = review.loc[scoped.path, "reviewed"].values
        scoped = scoped[~reviewed_mask]
    if max_confidence < 1.0:
        scoped = scoped[scoped.confidence.astype(float) < max_confidence]

    if per_class_n is not None:
        scoped = scoped.assign(_conf=scoped.confidence.astype(float)).sort_values("_conf")
        scoped = scoped.groupby("class", group_keys=False).head(per_class_n)

    if sort_uncertain_first:
        scoped = scoped.assign(_conf=scoped.confidence.astype(float)).sort_values("_conf")

    queue = scoped.path.tolist()

    st.sidebar.metric("Images in this scope", len(queue))

    total = len(review)
    done = int(review.reviewed.sum())
    st.sidebar.metric("Dataset-wide progress", f"{done} / {total}", f"{done / total:.1%}")

    if not queue:
        st.success("Nothing left to review in this scope.")
        return

    if "pos" not in st.session_state or st.session_state.pos >= len(queue):
        st.session_state.pos = 0
    pos = st.session_state.pos
    path = queue[pos]
    row = preds[preds.path == path].iloc[0]

    st.progress((pos + 1) / len(queue), text=f"{pos + 1} / {len(queue)} in this scope")
    declared_parts = sorted(declared.get(row["class"], []))
    st.caption(f"class: **{row['class']}**   declared affected part: "
               f"{', '.join(fmt_label(p) for p in declared_parts) or 'n/a'}")

    st.image(image_url(path), use_container_width=True)

    pred_label = predicted_label(row)
    st.markdown(f"### Predicted: `{fmt_label(pred_label)}`  "
                f"(confidence {float(row.confidence):.2f}, "
                f"2nd choice: `{fmt_label(row.second_organ)}` {float(row.second_confidence):.2f})")

    def commit(label):
        save_one(path, row["class"], row.pred_organ, row.confidence, label)
        review.loc[path, ["final_label", "reviewed"]] = [label, True]
        st.session_state.pos = min(pos + 1, len(queue))
        st.rerun()

    if st.button(f"Correct — it is '{DISPLAY.get(pred_label, pred_label)}'",
                 type="primary", use_container_width=True):
        commit(pred_label if pred_label in LABELS else DEFAULT_LABEL)

    st.write("Wrong — set the correct part:")
    for group, parts in TAXONOMY.items():
        if len(parts) > 1:
            st.caption(group)
        cols = st.columns(max(len(parts), 1))
        for col, label in zip(cols, parts):
            if col.button(DISPLAY[label], key=f"btn_{label}", use_container_width=True):
                commit(label)

    nav1, nav2 = st.columns(2)
    if nav1.button("Skip", use_container_width=True):
        st.session_state.pos = min(pos + 1, len(queue))
        st.rerun()
    if nav2.button("Previous", use_container_width=True, disabled=pos == 0):
        st.session_state.pos = max(pos - 1, 0)
        st.rerun()


def main():
    if not st.secrets.get("IMAGE_PUBLIC_BASE") or not st.secrets.get("DB_URL"):
        st.error("Missing secrets: set IMAGE_PUBLIC_BASE and DB_URL in this app's Settings → Secrets. "
                 "See README.md for how to get them.")
        st.stop()

    preds = load_predictions()
    declared = load_declared()

    mode = st.sidebar.radio("Mode", ["Gallery (browse)", "Review (correct)"])
    st.sidebar.divider()
    if mode == "Gallery (browse)":
        gallery_mode(preds, declared)
    else:
        review_mode(preds, declared)


if __name__ == "__main__":
    main()
