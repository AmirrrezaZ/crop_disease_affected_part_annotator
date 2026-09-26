# What this is and why (plain version)

You're an ML engineer, not a web dev, so here's the deployment stack in
terms that should map onto things you already know.

## The problem

The original `annotate_organs_app.py` is a Streamlit script that reads
images off your local disk and writes corrections to a local CSV. That
only works while your machine is on and you're sitting at it. You wanted
to annotate from your phone, anytime, without your desktop running.

## The three pieces

Think of it like separating a training job from its data and its logs:

| Piece | Role | ML analogy |
|---|---|---|
| **Streamlit Community Cloud** | Runs `app.py` 24/7, gives it a public URL | Like a managed notebook host (Colab/SageMaker) that keeps a script alive instead of you running `streamlit run` in a terminal |
| **Supabase Storage** | Holds the 22,741 images (downscaled to ~384px, ~0.5GB) | An S3 bucket, specifically — literally speaks the S3 API, we just point `boto3` at Supabase's endpoint instead of AWS's |
| **Supabase Postgres** | Holds the annotation table (`organ_review`) | Your `organ_review.csv`, except it's a real database table so multiple sessions/devices can read and write it concurrently without file-lock issues |

`app.py` itself is basically unchanged logic from the local version —
same gallery/review modes, same "which organ is this" UI. Only the I/O
layer changed: image paths → HTTPS URLs, CSV read/write → SQL
read/write.

## Why two Supabase things and not one

Supabase is a hosting company that bundles a Postgres database + an S3-
compatible file store + a few other things under one account/project. We
originally used Cloudflare for the images and Supabase for the DB, then
you asked to consolidate — now it's all one account, just two different
*features* of that one account (like how a single S3 bucket and an RDS
instance are both "AWS" but are different services under one account).

## Credentials involved (three separate pairs, this is the annoying part)

1. **Storage (S3-compatible) credentials** — `SUPABASE_S3_ACCESS_KEY_ID`
   / `SUPABASE_S3_SECRET_ACCESS_KEY`. Used by `scripts/upload_images.py`
   to push image files. **Done** — working, upload is running.
2. **Database connection string** — `DB_URL`, a single `postgresql://...`
   URL with your DB password baked in. Used by `scripts/seed_database.py`
   and by the deployed app itself. **Blocked right now** — see below.
3. **Streamlit Cloud secrets** — later, you'll paste `DB_URL` and
   `IMAGE_PUBLIC_BASE` (the public image URL prefix) into the deployed
   app's Settings → Secrets, once it's created on share.streamlit.io.
   **Not started yet.**

## Current status (as of this doc)

- [x] `app.py` written, tested locally against a throwaway SQLite DB and
      against the real Supabase Storage bucket.
- [x] Code pushed to `github.com:AmirrrezaZ/crop_disease_affected_part_annotator`.
- [x] Supabase bucket `crop_disease_annotator` created, made public, and
      the full 22,741-image upload is running in the background right now
      (resumable — safe to re-run if it dies partway).
- [ ] **Blocked**: I need your database connection string to seed the
      `organ_review` table and to give the deployed app read/write
      access. I tried guessing the pooler hostname (it encodes an AWS
      region, e.g. `aws-0-eu-west-1...`) against ~15 candidate regions
      and all failed the same way — wrong region guess, not a wrong
      password. This isn't something I can brute-force sensibly; it's
      one copy-paste from your dashboard. See "what I need from you"
      below.
- [ ] Not started: seeding the DB, creating the Streamlit Cloud app,
      pasting secrets in, final smoke test of the live URL.

## What I need from you, right now

Supabase dashboard → your project → **Project Settings → Database →
Connection string → URI tab** → make sure the **Transaction pooler**
option is selected → copy the whole string. It looks like:

```
postgresql://postgres.ihzihyosmsvrgmlafobf:[YOUR-PASSWORD]@aws-0-<something>.pooler.supabase.com:6543/postgres
```

Paste me that whole line (with your real password already in it, or
I'll substitute the one you gave me earlier) and I'll take it from
there — no more guessing.
