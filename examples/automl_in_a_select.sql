
DROP TABLE IF EXISTS leads;
CREATE TABLE leads (
    id               serial PRIMARY KEY,
    lead_score       int,
    pages_viewed     int,
    employees        int,
    demo_booked      int,
    prior_purchases  int,
    converted        int
);

INSERT INTO leads (lead_score, pages_viewed, employees, demo_booked, prior_purchases, converted) VALUES
    (85,12,200,1,2, 1), (20,2, 50,0,0, 0), (90,15,500,1,3, 1), (15,1, 30,0,0, 0),
    (75,10,150,1,1, 1), (30,3, 80,0,0, 0), (95,18,800,1,4, 1), (10,1, 20,0,0, 0),
    (60,8, 100,0,1, 1), (88,14,300,1,2, 1), (18,2, 60,0,0, 0), (70,9, 120,1,1, 1),
    (12,1, 25,0,0, 0), (92,16,600,1,3, 1), (22,3, 70,0,0, 0), (65,7, 90,0,1, 1),
    (28,2, 45,0,0, 0),
    (80,11,180,1,2, NULL), (17,2, 35,0,0, NULL), (55,6, 95,0,1, NULL);

WITH support AS (
    SELECT array_agg(lead_score      ORDER BY id) AS ls,
           array_agg(pages_viewed    ORDER BY id) AS pv,
           array_agg(employees       ORDER BY id) AS em,
           array_agg(demo_booked     ORDER BY id) AS db,
           array_agg(prior_purchases ORDER BY id) AS pp,
           array_agg(converted       ORDER BY id) AS y
    FROM leads WHERE converted IS NOT NULL
),
sup_flat AS (
    SELECT (SELECT array_agg(v) FROM (
                SELECT unnest(ARRAY[ls[i], pv[i], em[i], db[i], pp[i]])
                FROM generate_subscripts(ls,1) AS i
            ) t(v)) AS feats,
           y AS labels
    FROM support
),
query AS (
    SELECT id, ARRAY[lead_score, pages_viewed, employees, demo_booked, prior_purchases]::float8[] AS qf
    FROM leads WHERE converted IS NULL
)
SELECT q.id,
       fela_argmax(p.probs)     AS predicted_converted,
       round(fela_confidence(p.probs)::numeric, 3) AS confidence,
       ARRAY(SELECT round(x::numeric,3) FROM unnest(p.probs) x) AS class_probs
FROM query q, sup_flat s,
     LATERAL (SELECT fela_classify(q.qf, s.feats, s.labels, 5, 2) AS probs) p
ORDER BY q.id;

WITH support AS (
    SELECT array_agg(lead_score      ORDER BY id) AS ls,
           array_agg(pages_viewed    ORDER BY id) AS pv,
           array_agg(employees       ORDER BY id) AS em,
           array_agg(demo_booked     ORDER BY id) AS db,
           array_agg(prior_purchases ORDER BY id) AS pp,
           array_agg(converted       ORDER BY id) AS y
    FROM leads WHERE converted IS NOT NULL
),
sup_flat AS (
    SELECT (SELECT array_agg(v) FROM (
                SELECT unnest(ARRAY[ls[i], pv[i], em[i], db[i], pp[i]])
                FROM generate_subscripts(ls,1) AS i
            ) t(v)) AS feats, y AS labels
    FROM support
)
SELECT l.id,
       round(fela_confidence(fela_classify(
           ARRAY[l.lead_score,l.pages_viewed,l.employees,l.demo_booked,l.prior_purchases]::float8[],
           s.feats, s.labels, 5, 2))::numeric, 3) AS confidence,
       (fela_confidence(fela_classify(
           ARRAY[l.lead_score,l.pages_viewed,l.employees,l.demo_booked,l.prior_purchases]::float8[],
           s.feats, s.labels, 5, 2)) < 0.55) AS flagged_low_confidence
FROM leads l, sup_flat s
WHERE l.converted IS NULL
ORDER BY confidence ASC;

DROP TABLE IF EXISTS homes;
CREATE TABLE homes (
    id            serial PRIMARY KEY,
    beds          int,
    baths         int,
    sqft          int,
    age           int,
    garage        int,
    dist_downtown float8,
    price_usd     int
);

INSERT INTO homes (beds, baths, sqft, age, garage, dist_downtown, price_usd) VALUES
    (2,1, 900,35,0,12, 197000), (3,2,1400,20,1, 8, 338000), (4,3,2200,10,1, 5, 514000),
    (3,2,1800,15,1,10, 409000), (5,4,3200, 5,1, 3, 719000), (2,1,1000,40,0,15, 201000),
    (3,2,1600,25,0, 7, 361000), (4,2,2000, 8,1, 6, 471000), (6,5,3500, 2,1, 4, 786000),
    (3,1,1300,45,0,18, 250000), (4,3,2400,12,1, 9, 536000), (3,2,1750,18,1,11, 394000),
    (5,3,2900, 7,1, 2, 660000), (2,2,1100,22,0,13, 247000), (4,2,2100,14,1, 8, 477000),
    (3,2,1900, 6,1,10, 435000), (6,4,3300, 3,1, 5, 740000),
    (3,2,1650,16,1, 9, NULL), (5,3,2800, 9,1, 4, NULL), (2,1,1050,28,0,14, NULL);

WITH support AS (
    SELECT array_agg(beds::float8          ORDER BY id) AS bd,
           array_agg(baths::float8         ORDER BY id) AS ba,
           array_agg(sqft::float8          ORDER BY id) AS sf,
           array_agg(age::float8           ORDER BY id) AS ag,
           array_agg(garage::float8        ORDER BY id) AS gr,
           array_agg(dist_downtown         ORDER BY id) AS dd,
           array_agg(price_usd::float8     ORDER BY id) AS y
    FROM homes WHERE price_usd IS NOT NULL
),
sup_flat AS (
    SELECT (SELECT array_agg(v) FROM (
                SELECT unnest(ARRAY[bd[i], ba[i], sf[i], ag[i], gr[i], dd[i]])
                FROM generate_subscripts(bd,1) AS i
            ) t(v)) AS feats,
           y AS labels
    FROM support
),
query AS (
    SELECT id, ARRAY[beds, baths, sqft, age, garage, dist_downtown]::float8[] AS qf
    FROM homes WHERE price_usd IS NULL
)
SELECT q.id,
       round(r.out[1]::numeric, 0) AS predicted_price_usd,
       round(r.out[2]::numeric, 0) AS band_std
FROM query q, sup_flat s,
     LATERAL (SELECT fela_regress(q.qf, s.feats, s.labels, 6) AS out) r
ORDER BY q.id;

WITH support AS (
    SELECT array_agg(beds::float8          ORDER BY id) AS bd,
           array_agg(baths::float8         ORDER BY id) AS ba,
           array_agg(sqft::float8          ORDER BY id) AS sf,
           array_agg(age::float8           ORDER BY id) AS ag,
           array_agg(garage::float8        ORDER BY id) AS gr,
           array_agg(dist_downtown         ORDER BY id) AS dd,
           array_agg(price_usd::float8     ORDER BY id) AS y
    FROM homes WHERE price_usd IS NOT NULL
),
sup_flat AS (
    SELECT (SELECT array_agg(v) FROM (
                SELECT unnest(ARRAY[bd[i], ba[i], sf[i], ag[i], gr[i], dd[i]])
                FROM generate_subscripts(bd,1) AS i
            ) t(v)) AS feats,
           y AS labels
    FROM support
)
SELECT h.id, h.price_usd,
       round(r.out[1]::numeric, 0) AS gbm_predicted,
       round(abs(h.price_usd - r.out[1])::numeric, 0) AS abs_residual
FROM homes h, sup_flat s,
     LATERAL (SELECT fela_regress(
         ARRAY[h.beds,h.baths,h.sqft,h.age,h.garage,h.dist_downtown]::float8[],
         s.feats, s.labels, 6) AS out) r
WHERE h.price_usd IS NOT NULL
ORDER BY abs_residual DESC
LIMIT 5;

