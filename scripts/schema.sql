-- Run once against the Supabase Postgres database (SQL editor in the
-- Supabase dashboard, or `psql "$DB_URL" -f scripts/schema.sql`).
create table if not exists organ_review (
    path         text primary key,
    class        text not null,
    pred_organ   text,
    confidence   text,
    final_label  text,
    reviewed     boolean not null default false,
    reviewed_at  timestamptz
);

create index if not exists organ_review_reviewed_idx on organ_review (reviewed);
create index if not exists organ_review_class_idx on organ_review (class);
