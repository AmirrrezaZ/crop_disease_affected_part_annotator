# Crop Disease — Symptom Annotator

A small web app for annotating **disease symptoms** in a plant-disease photo
dataset: the **affected part** (leaf, stem, fruit, flower, root, seed/grain,
tuber, bulb, or whole plant) and the symptom attributes **color, texture,
shape and pattern** — optionally with GPT suggestions.

## 🌐 Live Web App

The annotator is available online and can be used directly from a browser:

**https://disease-annotator.streamlit.app/**

No local installation is required to use the deployed application.

## Why this exists

It's a supporting tool for a larger multimodal crop-disease diagnosis
project. That project's vision model is trained per-disease-class on
symptom attributes — including *affected_part* — but the images
themselves were never labeled with which plant organ they actually show.
A separate lightweight image classifier (CLIP embeddings + a small MLP
head) predicts an organ for every image, but like any model it's
imperfect, and its errors are exactly the images most likely to be
mislabeled in the source dataset.

This app closes that loop: it shows the model's prediction next to each
image, a human confirms or corrects it, and those corrections both (a)
fix the disease dataset's labels and (b) become new training data to
recalibrate the organ classifier itself.

## How it works

- **Gallery mode** — a grid of thumbnails for a chosen predicted organ
  (e.g. "show me everything predicted `root`"), optionally filtered to
  one disease class or to low-confidence predictions. Pick the right
  label from the dropdown under an image and it saves immediately — no
  extra click — and drops out of the grid once reviewed.
- **Review mode** — one image at a time, for working through a specific
  queue (e.g. every class where the model's majority prediction disagrees
  with that class's documented affected part).
- Every correction is written to a shared database immediately, so
  progress is the same whether you're on a laptop or a phone, and
  multiple sessions never clobber each other.

## Class-level labels from a vision LLM

For classes where the old single label is too coarse for the new taxonomy
(potato rot: root → tuber, corn smut: flower → inflorescence, ...),
a local-only script (`scripts/predict_class_parts.py`, gitignored — it is
not part of this repo) samples 5 random images per class, asks
`gpt-5-mini` (via AvalAI) which part each shows, and applies the **mode** to
every image of that class.

- `data/class_part_candidates.csv` — the hand-checked list of classes to
  process, with the reason for each (edit it to add/remove classes).
- `data/class_part_samples.csv` — every sampled image and the model's answer.
- `data/class_part_labels.csv` — the winning label per class. The app uses it
  to override the per-image CLIP prediction for those classes.
- Ties (fewer than 3/5 votes) get one extra round of 5 images; still-unclear
  classes are flagged `needs_check`.
- `--apply` / `--apply-only` writes the label to `organ_review` with
  `label_source='gpt_class'`, **only for images no human has reviewed yet**.

```bash
python3.10 -m venv .venv && .venv/bin/pip install openai boto3 psycopg2-binary pandas pillow tqdm python-dotenv requests
# .env needs AVALAI_API_KEY (and DB_URL for --apply)
.venv/bin/python scripts/predict_class_parts.py --list-classes
.venv/bin/python scripts/predict_class_parts.py            # label, no DB writes
.venv/bin/python scripts/predict_class_parts.py --apply    # also update the DB
.venv/bin/python scripts/check_bucket.py --http-sample 40  # verify every image is in the bucket
```

**Per-image, multi-part labels.** For the single-part, non-leaf diseased
classes (corn smut, wheat head scab, potato/onion/garlic rot, banana/blueberry
fruit diseases, ... — 18 classes, 1,396 images) a second local script,
`scripts/predict_image_parts.py`, judges every image on its own and may return
several parts (e.g. `leaf;fruit`). Results are in `data/image_part_labels.csv`
and take priority over the class-level vote in the app. A label with several
parts is stored `;`-joined in `organ_review.final_label`, and the gallery /
review screens let you select more than one part per image.

## Healthy crops

Classes with `disease_type == healthy` (celery leaf, corn leaf, coffee leaf, ...) have
no diseased part, so their organ label is `none` ("None (healthy)"). The app predicts
`none` for them and leaves them out of the flagged queue; to write it to the database
(marked reviewed, `label_source = 'class_rule'`):

```bash
DB_URL=... python scripts/set_healthy_organ.py            # dry run
DB_URL=... python scripts/set_healthy_organ.py --apply    # add --keep-human to spare human edits
```

## Where annotations are saved, and how to pull them

Every change is written straight to the Supabase Postgres database (`DB_URL`):
`organ_review` (affected part) and `attribute_review` (color / texture / shape /
pattern). Human edits have `label_source = 'human'`, `reviewed = TRUE`. In the
Attributes mode you can also type your own value (Enter to add it); it is stored
lower-cased with underscores and offered for later images.

```bash
pip install pandas sqlalchemy "psycopg[binary]"
DB_URL="postgresql+psycopg://..." python scripts/export_annotations.py        # human rows only -> exports/*.csv
DB_URL=... python scripts/export_annotations.py --all                         # everything
```

Use the transaction-pooler `DB_URL`; the app disables psycopg3 prepared
statements for it (otherwise `prepared statement "_pg3_0" does not exist`).

## Symptom attributes (color, texture, shape, pattern)

Besides *affected part*, each image also has `color`, `texture`, `shape` and
`pattern` attributes, stored in the `attribute_review` table (one row per image
and attribute, `;`-joined when several apply). `scripts/seed_attributes.py`
fills it from `data/class_attributes.csv` with these rules:

| Class situation | Result |
|---|---|
| Healthy class | `none`, marked reviewed |
| Diseased, attribute has no value | `none`, marked reviewed |
| Diseased, attribute has exactly **one** value | that value for every image of the class, marked reviewed |
| Diseased, attribute has **2+ values** | left for annotation — pick which of *that class's* values are visible |

Images still to annotate: color 14,733 · shape 7,751 · pattern 9,167 ·
texture 4,672 (14,995 distinct images; most need several attributes).

The app's **Attributes** mode shows one multiselect per image, limited to the
class's own values (plus `none`), saves on change, and has a progress bar per
attribute. Suggestions from GPT appear pre-filled with a "GPT suggestion" badge
and still count as *unreviewed* until a person confirms them.

### GPT-assisted annotation (Attributes mode)

Each card has a **🤖 Ask GPT** button (and there is an "Ask GPT for all shown"
button). GPT looks at the image, picks from the class's allowed values and may
also **propose its own label**. The result only fills the chips — nothing is
saved until you press Confirm.

**API key handling** — the key is never in the repo:
- *Deployed app owner*: put `AVALAI_API_KEY` (or `OPENAI_API_KEY`) in the app's
  **Settings → Secrets** (locally: `.streamlit/secrets.toml`, gitignored).
  Optional: `AVALAI_BASE_URL`, `GPT_MODEL` (default `gpt-5-mini`). Everyone using
  the app then gets the buttons without seeing the key. Note this spends your quota.
- *Otherwise*: each user pastes their own key into the sidebar field
  (password box, kept only in that browser session, not stored).

The app's progress panel counts only human-reviewed images as "reviewed";
GPT-labelled-but-unchecked images are shown as a separate number.

## Stack

- **UI**: [Streamlit](https://streamlit.io), deployed on Streamlit
  Community Cloud.
- **Images**: Supabase Storage, downscaled to ~384px on upload (organ
  identification doesn't need full resolution, and it keeps the full
  ~22.7k-image dataset under 0.5GB).
- **Annotations**: a Postgres table (Supabase), so corrections persist
  centrally instead of in a local CSV.

`data/organ_predictions.csv` and `data/class_attributes.csv` are the only
things committed to this repo (small metadata, a few MB) — the actual
images live in object storage, not in git.

---

## Deploying your own copy

### 1. Create a Supabase project

[supabase.com](https://supabase.com) → **New project**.

### 2. Database (annotation storage)

1. Project → **SQL Editor** → paste and run `scripts/schema.sql`.
2. Project → **Settings** → **Database** → **Connection string** → **URI**
   tab → copy the **Transaction pooler** string (port 6543). This is
   `DB_URL`. Replace `[YOUR-PASSWORD]` with your project's DB password.
3. Seed it with the current predictions (and carry over any existing
   human reviews from the original local `organ_review.csv`):
   ```bash
   pip install -r scripts/requirements-migration.txt
   DB_URL="postgresql://..." python scripts/seed_database.py
   ```

### 3. Storage (images)

1. Project → **Storage** → **New bucket** → toggle **Public bucket**.
2. Project → **Settings** → **Storage** → **S3 Connection** → **New
   access key**. Note the endpoint URL, region, access key ID, and secret.
3. Upload the dataset (run locally, where the full-res images already are
   — this resizes them on the way up):
   ```bash
   SUPABASE_S3_ENDPOINT=https://<project-ref>.storage.supabase.co/storage/v1/s3 \
   SUPABASE_S3_REGION=<region-from-that-page> \
   SUPABASE_S3_ACCESS_KEY_ID=... \
   SUPABASE_S3_SECRET_ACCESS_KEY=... \
   SUPABASE_BUCKET=<your-bucket-name> \
     python scripts/upload_images.py
   ```
   Uploads ~22.7k resized images with 12 parallel workers; skips files
   already present, so it's safe to re-run if interrupted. Add
   `--max-dim 512` for sharper thumbnails (~0.81GB instead of ~0.48GB).
4. `IMAGE_PUBLIC_BASE` is:
   `https://<project-ref>.supabase.co/storage/v1/object/public/<your-bucket-name>`

### 4. Streamlit Community Cloud (hosting)

The current deployed application is:

**https://disease-annotator.streamlit.app/**

To deploy your own copy:

1. Push this repo to GitHub.
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick
   this repo, branch `main`, main file `app.py`.
3. App **Settings** → **Secrets** → paste `DB_URL` and `IMAGE_PUBLIC_BASE`
   (see `.streamlit/secrets.toml.example`).
4. Deploy. Your app will be available at:
   `https://<your-app>.streamlit.app`

### Local testing against the real DB/bucket

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in your values
streamlit run app.py
```

### Syncing corrections back to the main project

The Supabase `organ_review` table is the source of truth for annotations.
To pull it back into the original project's `data/organ_review.csv` for
retraining the organ classifier:

```bash
psql "$DB_URL" -c "\copy organ_review TO 'organ_review.csv' WITH CSV HEADER"
```