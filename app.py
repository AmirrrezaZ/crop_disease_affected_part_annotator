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
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError, ProgrammingError

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PREDICTIONS = DATA / "organ_predictions.csv"
CLASS_ATTRS = DATA / "class_attributes.csv"
CLASS_PART_LABELS = DATA / "class_part_labels.csv"  # per-class GPT vote
IMAGE_PART_LABELS = DATA / "image_part_labels.csv"  # per-image GPT labels (may be multi-part)

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


def parse_labels(value) -> list:
    """'leaf;fruit' -> ['leaf', 'fruit'] (valid labels only, canonical order, no dupes).
    An image can show several affected parts; they are stored ';'-joined, the
    same convention as class_attributes.csv."""
    if not isinstance(value, str):
        return []
    found = {norm_label(v) for v in value.replace(",", ";").split(";")}
    return [lab for lab in LABELS if lab in found]


def join_labels(labels) -> str:
    return ";".join(parse_labels(";".join(labels)))


def fmt_label(label: str) -> str:
    """'Reproductive › Fruit' style display text; multi-part -> 'Leaf + Fruit'."""
    if isinstance(label, str) and ";" in label:
        return " + ".join(fmt_label(p) for p in parse_labels(label))
    lab = norm_label(label)
    if lab in GROUP_OF and GROUP_OF[lab] != DISPLAY[lab]:
        return f"{GROUP_OF[lab]} › {DISPLAY[lab]}"
    return DISPLAY.get(lab, label or "—")


st.set_page_config(page_title="Organ annotation", layout="centered")


def image_url(path: str) -> str:
    return f"{IMAGE_PUBLIC_BASE}/{path}"


def engine_kwargs(db_url: str) -> dict:
    """create_engine kwargs for this DB_URL's driver. Supabase's transaction
    pooler (port 6543) hands every statement to a different backend, so
    psycopg3's server-side prepared statements ("_pg3_0 does not exist") must
    be off. pre_ping/recycle drop connections the pooler has silently closed."""
    kwargs = {"pool_pre_ping": True, "pool_recycle": 300}
    if make_url(db_url).get_driver_name() == "psycopg":  # psycopg3 (not psycopg2)
        kwargs["connect_args"] = {"prepare_threshold": None}
    return kwargs


@st.cache_resource
def get_conn():
    db_url = st.secrets["DB_URL"]
    return st.connection("organ_db", type="sql", url=db_url, **engine_kwargs(db_url))


def run_write(sql: str, params: dict):
    """Execute one write; on a dropped/stale pooled connection, throw the pool
    away and retry (a few times) instead of surfacing the error to the user."""
    for attempt in range(3):
        conn = get_conn()
        try:
            with conn.session as s:
                s.execute(text(sql), params=params)
                s.commit()
            return
        except (OperationalError, InterfaceError, ProgrammingError, DBAPIError):
            conn.engine.dispose()
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))


@st.cache_data
def load_declared():
    declared = {}
    if CLASS_ATTRS.exists():
        df = pd.read_csv(CLASS_ATTRS, dtype=str).fillna("")
        for _, r in df.iterrows():
            declared[r["class"]] = set(norm_label(v) for v in r["affected_part"].split(";") if v)
    return declared


@st.cache_data
def load_class_labels():
    """class -> part chosen by the GPT 5-sample vote (scripts/predict_class_parts.py).
    These replace the per-image CLIP prediction for the whole class, because
    the CLIP organ labels are too coarse for those classes (potato: root ->
    tuber, corn smut: flower -> inflorescence, ...)."""
    if not CLASS_PART_LABELS.exists():
        return {}
    df = pd.read_csv(CLASS_PART_LABELS, dtype=str).fillna("")
    df = df[df["label"] != ""]
    return {r["class"]: norm_label(r["label"]) for _, r in df.iterrows()}


@st.cache_data
def load_image_labels():
    """path -> ';'-joined parts from the per-image GPT pass (scripts/predict_image_parts.py)."""
    if not IMAGE_PART_LABELS.exists():
        return {}
    df = pd.read_csv(IMAGE_PART_LABELS, dtype=str).fillna("")
    df["parts"] = df["parts"].map(lambda v: join_labels(parse_labels(v)))
    return {r["path"]: r["parts"] for _, r in df.iterrows() if r["parts"]}


@st.cache_data
def load_predictions():
    """Effective prediction per image, highest priority first:
    per-image GPT labels > per-class GPT vote > the CLIP organ head."""
    df = pd.read_csv(PREDICTIONS, dtype=str).fillna("")
    df["pred_affected_part"] = df["pred_affected_part"].map(norm_label)
    df["model_part"] = df["pred_affected_part"]  # the image-level CLIP prediction, untouched
    class_override = df["class"].map(load_class_labels())
    image_override = df["path"].map(load_image_labels())
    df["gpt_label"] = class_override.notna() | image_override.notna()
    df["pred_affected_part"] = (image_override.fillna(class_override)
                                .fillna(df["pred_affected_part"]))
    return df


def predicted_label(row) -> str:
    return row.pred_affected_part or norm_label(row.pred_organ) or DEFAULT_LABEL


def flagged_classes(preds, declared):
    """Classes whose majority-predicted organ isn't in the declared affected_part set."""
    flagged = []
    for cls, sub in preds[preds.pred_organ != ""].groupby("class"):
        majority_part = sub.model_part.mode()
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
    review["final_label"] = review["final_label"].map(
        lambda v: join_labels(parse_labels(v)) or norm_label(v))

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
    run_write("""
        INSERT INTO organ_review (path, class, pred_organ, confidence, final_label, reviewed, reviewed_at, label_source)
        VALUES (:path, :cls, :pred_organ, :confidence, :final_label, TRUE, :reviewed_at, 'human')
        ON CONFLICT (path) DO UPDATE SET
            final_label = EXCLUDED.final_label,
            reviewed = TRUE,
            reviewed_at = EXCLUDED.reviewed_at,
            label_source = 'human'
    """, {"path": path, "cls": cls, "pred_organ": pred_organ, "confidence": confidence,
          "final_label": final_label, "reviewed_at": datetime.now(timezone.utc)})


def progress_panel(preds, review):
    """Overall review progress, shown on top of every mode.

    Human-reviewed = rows with reviewed=TRUE (a person confirmed/corrected it).
    Class-level GPT = still-unreviewed images whose label came from the
    5-sample GPT vote for their class; they are labelled but nobody has
    looked at them yet, so they are tracked separately, not as 'reviewed'."""
    total = len(review)
    done = int(review.reviewed.sum())
    ok_paths = review.index.intersection(preds.loc[preds.gpt_label, "path"])
    gpt_pending = int((~review.loc[ok_paths, "reviewed"]).sum()) if len(ok_paths) else 0
    pct = done / total if total else 0.0

    st.progress(min(pct, 1.0), text=f"Reviewed: {done:,} / {total:,} images ({pct:.1%})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviewed", f"{done:,}")
    c2.metric("Left to review", f"{total - done:,}")
    c3.metric("GPT-labelled, unreviewed", f"{gpt_pending:,}",
              help="Images whose part(s) come from GPT and that nobody has checked yet.")

    with st.expander("Progress by class"):
        by_class = review.groupby("class").reviewed.agg(reviewed="sum", total="count")
        by_class["progress"] = (by_class.reviewed / by_class.total).round(3)
        st.dataframe(
            by_class.sort_values("progress"),
            column_config={"progress": st.column_config.ProgressColumn(
                "progress", min_value=0.0, max_value=1.0, format="percent")},
            use_container_width=True,
        )


def gallery_mode(preds, declared):
    """Grid sanity-check for a chosen predicted affected_part, with an
    inline correct-or-confirm control per thumbnail so you can fix wrong
    ones right from the grid instead of one image at a time."""
    st.title("Organ predictions — gallery")

    review = st.session_state.review
    progress_panel(preds, review)

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

    scoped = preds_ok[preds_ok.pred_affected_part.map(lambda s: part in parse_labels(s))]
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
        picked = st.session_state[f"sel_{path}"]
        if not picked:  # nothing selected: keep the previous label
            st.toast("Pick at least one part.")
            return
        sel = join_labels(picked)
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
            st.multiselect(
                "part(s)", LABELS, default=parse_labels(current_label) or [DEFAULT_LABEL],
                format_func=fmt_label,
                key=f"sel_{path}", label_visibility="collapsed",
                help="Select every part that visibly shows the disease.",
                on_change=on_label_change,
                args=(path, row["class"], row.pred_organ, row.confidence, only_unreviewed),
            )
            if not is_reviewed and st.button("✓ Confirm as-is", key=f"confirm_{path}",
                                              use_container_width=True):
                on_label_change(path, row["class"], row.pred_organ, row.confidence, only_unreviewed)
                st.rerun()


def review_mode(preds, declared):
    st.title("Organ annotation")

    review = st.session_state.review
    progress_panel(preds, review)

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

    if st.button(f"Correct — it is '{fmt_label(pred_label)}'",
                 type="primary", use_container_width=True):
        commit(join_labels(parse_labels(pred_label)) or DEFAULT_LABEL)

    st.write("Wrong — set the correct part (one tap):")
    for group, parts in TAXONOMY.items():
        if len(parts) > 1:
            st.caption(group)
        cols = st.columns(max(len(parts), 1))
        for col, label in zip(cols, parts):
            if col.button(DISPLAY[label], key=f"btn_{label}", use_container_width=True):
                commit(label)

    picked = st.multiselect(
        "Or several parts at once", LABELS, default=parse_labels(pred_label),
        format_func=fmt_label, key=f"multi_{path}",
        help="Select every part that visibly shows the disease, then save.")
    if st.button("Save selected parts", disabled=not picked, use_container_width=True):
        commit(join_labels(picked))

    nav1, nav2 = st.columns(2)
    if nav1.button("Skip", use_container_width=True):
        st.session_state.pos = min(pos + 1, len(queue))
        st.rerun()
    if nav2.button("Previous", use_container_width=True, disabled=pos == 0):
        st.session_state.pos = max(pos - 1, 0)
        st.rerun()


# ------------------------------------------------- symptom attributes
ATTRIBUTES = ["color", "texture", "shape", "pattern"]
NONE_VALUE = "none"  # healthy / not visible / not applicable


@st.cache_data
def load_attribute_vocab():
    """(vocab, per_class): vocab[attr] = every value used in class_attributes.csv,
    per_class[attr][class] = the values declared for that class."""
    df = pd.read_csv(CLASS_ATTRS, dtype=str).fillna("")
    vocab, per_class = {}, {}
    for a in ATTRIBUTES:
        vals = df[a].map(lambda s: [v for v in s.split(";") if v])
        vocab[a] = sorted({v for vs in vals for v in vs})
        per_class[a] = dict(zip(df["class"], vals))
    return vocab, per_class


def fmt_attr(value: str) -> str:
    return value.replace("_", " ")


def clean_attr(value) -> str:
    """Typed-in custom value -> vocabulary style: 'Dark  Brown;' -> 'dark_brown'."""
    return "_".join(str(value).replace(";", " ").replace(",", " ").lower().split())


def split_attr(value) -> list:
    if not isinstance(value, str):
        return []
    return [v for v in (clean_attr(x) for x in value.split(";")) if v]


def join_attr(values, vocab) -> str:
    """Canonical ';'-joined value: vocab values in vocab order, then custom
    (typed-in) values in the order given; 'none' only stands alone."""
    vals = list(dict.fromkeys(v for v in (clean_attr(x) for x in values) if v))
    if len(vals) > 1 and NONE_VALUE in vals:
        vals.remove(NONE_VALUE)
    known = [v for v in vocab if v in vals]
    return ";".join(known + [v for v in vals if v not in vocab])


def load_attribute_rows(attribute):
    key = f"attr_rows_{attribute}"
    if key not in st.session_state:
        df = get_conn().query(
            "SELECT path, class, value, reviewed, label_source FROM attribute_review "
            "WHERE attribute = :a", params={"a": attribute}, ttl=0)
        df["reviewed"] = df["reviewed"].astype(bool)
        df["value"] = df["value"].fillna("")
        df["label_source"] = df["label_source"].fillna("")
        st.session_state[key] = df.set_index("path")
    return st.session_state[key]


def save_attribute(path, attribute, cls, value):
    run_write("""
        INSERT INTO attribute_review (path, attribute, class, value, reviewed, reviewed_at, label_source)
        VALUES (:path, :attribute, :cls, :value, TRUE, :reviewed_at, 'human')
        ON CONFLICT (path, attribute) DO UPDATE SET
            value = EXCLUDED.value, reviewed = TRUE,
            reviewed_at = EXCLUDED.reviewed_at, label_source = 'human'
    """, {"path": path, "attribute": attribute, "cls": cls, "value": value,
          "reviewed_at": datetime.now(timezone.utc)})


def attribute_progress():
    """One progress bar per attribute (reviewed = a person or a class rule fixed it)."""
    df = get_conn().query(
        "SELECT attribute, COUNT(*) AS total, COUNT(*) FILTER (WHERE reviewed) AS done, "
        "COUNT(*) FILTER (WHERE NOT reviewed AND label_source = 'gpt_image') AS gpt "
        "FROM attribute_review GROUP BY attribute", ttl=5).set_index("attribute")
    tot, done = int(df.total.sum()), int(df.done.sum())
    st.progress(done / tot if tot else 0.0,
                text=f"All attributes: {done:,} / {tot:,} reviewed ({done / tot:.1%})")
    cols = st.columns(len(ATTRIBUTES))
    for col, a in zip(cols, ATTRIBUTES):
        if a in df.index:
            r = df.loc[a]
            col.metric(a.title(), f"{r.done / r.total:.0%}",
                       help=f"{int(r.gpt):,} of the unreviewed have a GPT suggestion")
            col.caption(f"{int(r.total - r.done):,} left")


def attribute_mode(preds):
    st.title("Symptom attributes")
    vocab, per_class = load_attribute_vocab()
    attribute_progress()
    st.caption("Single-value classes and healthy classes are already filled in and counted as "
               "reviewed. Multi-value classes need a per-image choice among **that class's own** "
               "values (or `none` if the symptom isn't visible).")

    st.sidebar.header("Attributes")
    attribute = st.sidebar.selectbox("Attribute", ATTRIBUTES, format_func=str.title)
    rows = load_attribute_rows(attribute)
    declared = per_class[attribute]
    multi = {c for c, v in declared.items() if len(v) > 1}

    include_auto = st.sidebar.checkbox("Include auto-labelled classes", value=False,
                                       help="Single-value / healthy classes (already reviewed).")
    pool = sorted(rows["class"].unique()) if include_auto else sorted(multi & set(rows["class"]))
    cls = st.sidebar.selectbox("Class", ["(all classes)"] + pool)
    only_unreviewed = st.sidebar.checkbox("Only unreviewed", value=True)
    only_gpt = st.sidebar.checkbox("Only GPT-suggested", value=False)
    show_all = st.sidebar.checkbox("Offer all values (not just the class's)", value=False)
    n = st.sidebar.slider("How many images", 1, 48, 12)
    cols_n = st.sidebar.slider("Grid columns", 1, 6, 2)

    scoped = rows[rows["class"].isin(pool)]
    if cls != "(all classes)":
        scoped = scoped[scoped["class"] == cls]
    if only_unreviewed:
        scoped = scoped[~scoped.reviewed]
    if only_gpt:
        scoped = scoped[scoped.label_source == "gpt_image"]
    st.sidebar.metric("Matching images", len(scoped))

    sig = (attribute, cls, only_unreviewed, only_gpt, n)
    resample = st.sidebar.button("🔄 New sample")
    if st.session_state.get("attr_sig") != sig or resample:
        sample = scoped.sample(min(n, len(scoped))) if len(scoped) else scoped
        st.session_state.attr_paths = sample.index.tolist()
        st.session_state.attr_sig = sig
    paths = st.session_state.get("attr_paths", [])
    if not paths:
        st.success("Nothing left in this selection." if only_unreviewed else "No images match.")
        return

    custom_seen = sorted({v for val in rows["value"] for v in split_attr(val)}
                         - set(vocab[attribute]) - {NONE_VALUE})

    def options_for(cls_val, current=()):
        base = vocab[attribute] if show_all else declared.get(cls_val, []) or vocab[attribute]
        extra = [v for v in list(custom_seen) + list(current) if v not in base]
        return list(dict.fromkeys(list(base) + [NONE_VALUE] + extra))

    def wkey(path):
        return f"attr_{attribute}_{int(show_all)}_{path}"

    def default_for(path):
        return split_attr(rows.loc[path, "value"])

    def commit(path):
        r = rows.loc[path]
        picked = st.session_state.get(wkey(path), default_for(path))
        if not picked:
            st.toast("Pick at least one value (or 'none').")
            return False
        value = join_attr(picked, vocab[attribute])
        save_attribute(path, attribute, r["class"], value)
        rows.loc[path, ["value", "reviewed", "label_source"]] = [value, True, "human"]
        st.session_state[wkey(path)] = value.split(";")  # show typed values in cleaned form
        # No removal from the grid here: the image stays on screen (marked
        # reviewed) so you can keep editing it; it drops on the next sample/filter change.
        return True

    st.caption(f"{len(paths)} images — choose every **{attribute}** value that is visibly part "
               "of the symptom; it saves immediately (the image counts as reviewed and stays "
               "here until the next sample, so you can keep editing). Type a new value and press "
               "Enter to add your own.")
    if st.button("✓ Confirm all shown (that have a selection)", use_container_width=True):
        for p in list(paths):
            if st.session_state.get(wkey(p), default_for(p)):  # skip images with nothing chosen
                commit(p)
        st.rerun()

    cols = st.columns(cols_n)
    for i, path in enumerate(paths):
        r = rows.loc[path]
        with cols[i % cols_n]:
            st.image(image_url(path), use_container_width=True)
            if r["reviewed"]:
                status = "✓ reviewed"
            elif r["label_source"] == "gpt_image":
                status = "GPT suggestion"
            else:
                status = "not annotated"
            st.caption(f"{r['class']} — {status}")
            st.caption("class values: " + (", ".join(fmt_attr(v) for v in declared.get(r["class"], [])) or "—"))
            st.multiselect(
                attribute, options_for(r["class"], default_for(path)), default=default_for(path),
                format_func=fmt_attr, key=wkey(path), label_visibility="collapsed",
                accept_new_options=True, on_change=commit, args=(path,),
            )
            if not r["reviewed"] and st.button("✓ Confirm", key=f"attrok_{attribute}_{path}",
                                               use_container_width=True):
                if commit(path):
                    st.rerun()


def main():
    if not st.secrets.get("IMAGE_PUBLIC_BASE") or not st.secrets.get("DB_URL"):
        st.error("Missing secrets: set IMAGE_PUBLIC_BASE and DB_URL in this app's Settings → Secrets. "
                 "See README.md for how to get them.")
        st.stop()

    preds = load_predictions()
    declared = load_declared()

    if "review" not in st.session_state:
        st.session_state.review = load_review_state(preds)
    mode = st.sidebar.radio("Mode", ["Gallery (browse)", "Review (correct)",
                                     "Attributes (color, texture, ...)"])
    st.sidebar.divider()
    if mode == "Gallery (browse)":
        gallery_mode(preds, declared)
    elif mode.startswith("Attributes"):
        attribute_mode(preds)
    else:
        review_mode(preds, declared)


if __name__ == "__main__":
    main()
