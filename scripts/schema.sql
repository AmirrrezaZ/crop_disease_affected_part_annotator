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

-- 'human' = confirmed/corrected in the app; 'gpt_class' = set from the per-class
-- GPT vote (scripts/predict_class_parts.py), not yet looked at by a person.
alter table organ_review add column if not exists label_source text;

create index if not exists organ_review_reviewed_idx on organ_review (reviewed);
create index if not exists organ_review_class_idx on organ_review (class);

-- The app connects with the postgres/pooler credential (table owner), which
-- always bypasses RLS -- this only blocks the separate public PostgREST API
-- Supabase auto-generates for every table, which the app never uses. With
-- RLS on and no policies, that API surface has zero access.
alter table organ_review enable row level security;

-- Per-image symptom attributes other than affected_part: color, texture, shape,
-- pattern (vocabularies come from data/class_attributes.csv). One row per
-- (image, attribute); `value` is ';'-joined for multi-valued answers, or 'none'
-- (healthy / attribute not applicable). label_source: 'class_rule' (class has
-- 0/1 values, or is healthy -> applied automatically and marked reviewed),
-- 'gpt_image' (GPT suggestion, unreviewed), 'human'.
create table if not exists attribute_review (
    path         text not null,
    attribute    text not null,
    class        text not null,
    value        text not null default '',
    reviewed     boolean not null default false,
    reviewed_at  timestamptz,
    label_source text,
    primary key (path, attribute)
);
create index if not exists attribute_review_attr_idx on attribute_review (attribute, reviewed);
create index if not exists attribute_review_attr_class_idx on attribute_review (attribute, class);
alter table attribute_review enable row level security;
