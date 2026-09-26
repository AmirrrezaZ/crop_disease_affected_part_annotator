# crop_disease affected-part annotator (deployed)

Deployed version of `crop_disease/version1`'s organ-annotation Streamlit
app: browse and correct per-image organ predictions (leaf/stem/fruit/
flower/root/seed_grain/whole_plant) from any device, any time — not tied
to whether the local machine is on.

Everything lives in **one Supabase project**:
- **Images**: Supabase Storage (a public bucket), downscaled to ~384px on
  upload — this app is for organ identification, not symptom diagnosis, so
  a thumbnail is plenty, and it keeps the whole 22.7k-image dataset at
  ~0.48GB, comfortably inside the 1GB free-tier storage cap.
- **Annotations**: a Postgres table in the same project, instead of a
  local CSV, so progress is shared across devices.
- **Layout**: tuned for phone-width viewports (2-column gallery by
  default, full-width buttons, auto-saves on label change); still usable
  on desktop.

`data/organ_predictions.csv` and `data/class_attributes.csv` are small
metadata files and are committed directly — only the actual images go
through Supabase Storage.

**Free-tier caveat**: an unused Supabase free project pauses after a week
of inactivity (data isn't lost, just needs a click to resume from the
dashboard before the app works again). Fine for occasional annotation
sessions; if you want it truly always-on with zero manual steps, that's
what the $25/mo Pro plan removes.

## One-time setup (~10 minutes)

### 1. Create the Supabase project

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

1. Project → **Storage** → **New bucket** (e.g. `crop-disease-images`) →
   toggle **Public bucket**.
2. Project → **Settings** → **Storage** → **S3 Connection** → **New
   access key**. Note the endpoint URL, region, access key ID, and secret.
3. Upload the dataset (run locally, where the full-res images already are
   — this resizes them on the way up):
   ```bash
   SUPABASE_S3_ENDPOINT=https://<project-ref>.supabase.co/storage/v1/s3 \
   SUPABASE_S3_REGION=<region-from-that-page> \
   SUPABASE_S3_ACCESS_KEY_ID=... \
   SUPABASE_S3_SECRET_ACCESS_KEY=... \
   SUPABASE_BUCKET=crop-disease-images \
     python scripts/upload_images.py
   ```
   Uploads ~22.7k resized images with 12 parallel workers; skips files
   already present, so it's safe to re-run if interrupted. Add
   `--max-dim 512` if you want sharper thumbnails and don't mind using
   closer to the full 1GB (~0.81GB at 512px).
4. `IMAGE_PUBLIC_BASE` is:
   `https://<project-ref>.supabase.co/storage/v1/object/public/crop-disease-images`

### 4. Streamlit Community Cloud (hosting)

1. Push this repo to GitHub (already done if you're reading this from
   there).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick
   this repo, branch `main`, main file `app.py`.
3. App **Settings** → **Secrets** → paste:
   ```toml
   DB_URL = "postgresql://...pooler.supabase.com:6543/postgres"
   IMAGE_PUBLIC_BASE = "https://xxxxxxxxxxxx.supabase.co/storage/v1/object/public/crop-disease-images"
   ```
   (see `.streamlit/secrets.toml.example`).
4. Deploy. The app is now live at `https://<your-app>.streamlit.app`,
   works from a phone browser, and stays up regardless of whether this
   machine is on (mod the free-tier pause note above).

## Local testing against the real DB/bucket

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in your values
streamlit run app.py
```

## Syncing corrections back to the main project

The Supabase `organ_review` table is now the source of truth for
annotations. To pull it back into `crop_disease/version1/data/organ_review.csv`
for retraining `plant_organ` (see that project's `scripts/build_manifest.py`,
`AFFECTED_PART_TO_ORGAN`), export the table:

```bash
psql "$DB_URL" -c "\copy organ_review TO 'organ_review.csv' WITH CSV HEADER"
```
