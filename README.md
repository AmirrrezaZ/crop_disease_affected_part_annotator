# crop_disease affected-part annotator (deployed)

Deployed version of `crop_disease/version1`'s organ-annotation Streamlit
app: browse and correct per-image organ predictions (leaf/stem/fruit/
flower/root/seed_grain/whole_plant) from any device, any time — not tied
to whether the local machine is on.

- **Images**: served from a Cloudflare R2 bucket instead of local disk.
- **Annotations**: read/written to a Supabase Postgres table instead of a
  local CSV, so progress is shared across devices.
- **Layout**: tuned for phone-width viewports (2-column gallery by
  default, full-width buttons); still usable on desktop.

`data/organ_predictions.csv` and `data/class_attributes.csv` are small
metadata files and are committed directly — only the actual images (4.2GB)
go through R2.

## One-time setup (~15 minutes)

### 1. Cloudflare R2 (image hosting, free up to 10GB)

1. Cloudflare dashboard → R2 → **Create bucket** (any name, e.g.
   `crop-disease-images`).
2. Bucket → **Settings** → **Public access** → enable the `r2.dev`
   subdomain (or attach a custom domain). Copy that public URL — this is
   `R2_PUBLIC_BASE`.
3. R2 → **Manage API tokens** → create a token with **Object Read & Write**
   permission on this bucket. Note the **Account ID**, **Access Key ID**,
   and **Secret Access Key**.
4. Upload the dataset (run locally, where the images already are):
   ```bash
   pip install -r scripts/requirements-migration.txt
   R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=crop-disease-images \
     python scripts/upload_images_to_r2.py
   ```
   Uploads ~22.7k files (4.2GB) with 16 parallel workers; skips files
   already present, so it's safe to re-run if interrupted.

### 2. Supabase (annotation database, free tier)

1. [supabase.com](https://supabase.com) → **New project**.
2. Project → **SQL Editor** → paste and run `scripts/schema.sql`.
3. Project → **Settings** → **Database** → **Connection string** → **URI**
   tab → copy the **Transaction pooler** string (port 6543). This is
   `DB_URL`. Replace `[YOUR-PASSWORD]` with your project's DB password.
4. Seed it with the current predictions (and carry over any existing
   human reviews from the original local `organ_review.csv`):
   ```bash
   DB_URL="postgresql://..." python scripts/seed_database.py
   ```

### 3. Streamlit Community Cloud (hosting)

1. Push this repo to GitHub (already done if you're reading this from
   there).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick
   this repo, branch `main`, main file `app.py`.
3. Before or after first deploy: app **Settings** → **Secrets** → paste:
   ```toml
   DB_URL = "postgresql://...pooler.supabase.com:6543/postgres"
   R2_PUBLIC_BASE = "https://pub-xxxxxxxxxxxxxxxx.r2.dev"
   ```
   (see `.streamlit/secrets.toml.example`).
4. Deploy. The app is now live at `https://<your-app>.streamlit.app`,
   works from a phone browser, and stays up regardless of whether this
   machine is on.

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
