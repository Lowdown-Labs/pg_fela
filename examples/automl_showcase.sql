
\echo '--- 0. introspection: version + caps + model info ---'
SELECT fela_version();
SELECT * FROM fela_caps();
SELECT fela_model_info();

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

DROP TABLE IF EXISTS salaries;
CREATE TABLE salaries (
    id         serial PRIMARY KEY,
    level      int,
    years_exp  int,
    city_tier  int,
    team_code  int,
    manages    int,
    salary_usd int
);
INSERT INTO salaries (level, years_exp, city_tier, team_code, manages, salary_usd) VALUES
    (1, 1,3,2,0,  78000), (3, 8,1,0,0, 143000), (2, 3,2,1,0, 109000), (5,20,1,0,1, 212000),
    (1, 0,3,2,0,  76000), (4,15,2,1,1, 177500), (2, 5,3,3,0, 103000), (3,10,1,2,0, 143000),
    (5,25,1,0,1, 219500), (1, 2,2,2,0,  87000), (4,12,3,0,1, 166000), (3, 7,2,3,0, 132000),
    (5,18,2,0,1, 201000), (1, 1,1,2,0,  94000), (4,14,1,1,1, 184000), (2, 6,3,2,0, 103000),
    (3, 9,2,0,0, 137000),
    (2, 5,2,1,0, NULL), (4,16,1,0,1, NULL), (1, 3,3,2,0, NULL);

\echo '--- 1a. AUTO task-type detection: homes -> regress ---'
SELECT fela_detect_task('homes','price_usd') AS task;

\echo '--- 1b. AutoML in a SELECT: predict the 3 NULL home prices (one call, pure GBM) ---'
SELECT * FROM fela_automl('homes','price_usd') ORDER BY row_id;

\echo '--- 1c. Direct fela_regress(...) array-primitive call: the same GBM path, called by hand ---'
WITH sup AS (
    SELECT array_agg(beds::float8          ORDER BY id) AS beds,
           array_agg(baths::float8         ORDER BY id) AS baths,
           array_agg(sqft::float8          ORDER BY id) AS sqft,
           array_agg(age::float8           ORDER BY id) AS age,
           array_agg(garage::float8        ORDER BY id) AS garage,
           array_agg(dist_downtown         ORDER BY id) AS dist,
           array_agg(price_usd::float8     ORDER BY id) AS price
    FROM homes WHERE price_usd IS NOT NULL
),
sup_flat AS (
    SELECT (SELECT array_agg(v) FROM (
                SELECT unnest(ARRAY[beds[i], baths[i], sqft[i], age[i], garage[i], dist[i]])
                FROM generate_subscripts(beds,1) AS i
            ) t(v)) AS feats,
           price AS labels
    FROM sup
)
SELECT round(r[1]::numeric,0) AS predicted_price_usd, round(r[2]::numeric,0) AS band_std
FROM sup_flat s,
     LATERAL (SELECT fela_regress(
         ARRAY[3,2,1700,16,1,9]::float8[],
         s.feats, s.labels, 6) AS r) x;

\echo '--- 1d. fela_conformal_regress: calibrated 80% prediction intervals for salaries ---'
SELECT fela_detect_task('salaries','salary_usd') AS task;
SELECT row_id, round(prediction::numeric,0) AS predicted,
       round(lo::numeric,0) AS lo, round(hi::numeric,0) AS hi
FROM fela_conformal_regress('salaries','salary_usd', 0.8) ORDER BY row_id;

\echo '--- 1e. Feature importance + plain-language explanation for the GBM regression on homes ---'
SELECT * FROM fela_importance('homes','price_usd') ORDER BY importance DESC;
SELECT fela_explain('homes','price_usd') AS explanation;

\echo '--- 1f. Anomaly / novelty on a regression target: standardized residual z-score (GBM, no model) ---'
SELECT * FROM fela_anomaly('homes','price_usd') ORDER BY score DESC;

\echo '--- 2a. AUTO task-type detection: leads -> classify ---'
SELECT fela_detect_task('leads','converted') AS task;

\echo '--- 2b. AutoML in a SELECT: predict the 3 NULL lead outcomes (one call, FelaTab) ---'
SELECT * FROM fela_automl('leads','converted') ORDER BY row_id;

\echo '--- 2c. Imputation: just the filled values for the NULL rows ---'
SELECT * FROM fela_impute('leads','converted') ORDER BY row_id;

\echo '--- 2d. Anomaly / novelty: per-labeled-row disagreement score (high = odd) ---'
SELECT * FROM fela_anomaly('leads','converted') ORDER BY score DESC;

\echo '--- 2e. Feature importance + plain-language explanation ---'
SELECT * FROM fela_importance('leads','converted') ORDER BY importance DESC;
SELECT fela_explain('leads','converted') AS explanation;

\echo '--- 2f. Conformal ABSTENTION threshold for classification (80% coverage) ---'
SELECT round(fela_conformal_threshold('leads','converted', 0.80)::numeric, 3) AS min_confidence;

\echo '--- 3. Clustering: k-means (k=3) on standardized home features ---'
DROP TABLE IF EXISTS home_features;
CREATE TABLE home_features AS SELECT beds, baths, sqft, age, garage, dist_downtown FROM homes;
SELECT cluster, count(*) FROM fela_cluster('home_features', 3) GROUP BY cluster ORDER BY cluster;

\echo '--- 4. Similarity search: nearest support homes to a query home (standardized distance) ---'
SELECT support_idx, round(distance::numeric,4) AS distance
FROM fela_similar(
       ARRAY[3,2,1700,16,1,9]::float8[],
       ARRAY[2,1, 900,35,0,12,  3,2,1400,20,1,8,  4,3,2200,10,1,5,
             3,2,1800,15,1,10,  5,4,3200, 5,1,3]::float8[],
       6, 3);

\echo '--- 5. Confidence gating: answer only when sure (else NULL) ---'
SELECT fela_classify_gated(ARRAY[80,11,180,1,2]::float8[],
         ARRAY[85,12,200,1,2, 20,2,50,0,0, 90,15,500,1,3,
               15,1,30,0,0,  75,10,150,1,1, 30,3,80,0,0]::float8[],
         ARRAY[1,0,1,0,1,0]::int[], 5, 2, 0.30) AS answer_lowthresh,
       fela_classify_gated(ARRAY[80,11,180,1,2]::float8[],
         ARRAY[85,12,200,1,2, 20,2,50,0,0, 90,15,500,1,3,
               15,1,30,0,0,  75,10,150,1,1, 30,3,80,0,0]::float8[],
         ARRAY[1,0,1,0,1,0]::int[], 5, 2, 0.99) AS answer_highthresh;

\echo '--- 6. IMPLICIT AutoML: one call builds a view with the ML columns joined back by PK ---'
SELECT fela_create_view('homes', 'price_usd');
SELECT id, price_usd, cluster, round(anomaly_score::numeric,0) AS anomaly,
       round(predicted::numeric,0) AS predicted, round(confidence::numeric,3) AS confidence,
       trust_label, ood, round(band::numeric,0) AS band
FROM homes_ml ORDER BY id;
REFRESH MATERIALIZED VIEW homes_ml;

SELECT fela_create_view('leads', 'converted');
SELECT id, converted, cluster, round(anomaly_score::numeric,3) AS anomaly,
       predicted, round(confidence::numeric,3) AS confidence, trust_label, ood, band
FROM leads_ml ORDER BY id;
REFRESH MATERIALIZED VIEW leads_ml;

\echo '--- 7. Inline _over() window functions: cluster/anomaly as column expressions ---'
SELECT id, price_usd,
       fela_cluster_over(beds, baths, sqft, age, garage, dist_downtown) OVER () AS cluster,
       round(fela_anomaly_over(beds, baths, sqft, age, garage, dist_downtown) OVER ()::numeric, 3) AS novelty
FROM homes ORDER BY id;

\echo '--- 8. FEVER embeddings: geometric ops on vectors (cluster / dedup / outliers) ---'
DROP TABLE IF EXISTS photos;
CREATE TABLE photos (id serial PRIMARY KEY, embedding float8[]);
INSERT INTO photos (embedding) VALUES
    (ARRAY[0.90,0.10,0.05,0.02]), (ARRAY[0.88,0.12,0.06,0.03]), (ARRAY[0.91,0.09,0.04,0.01]),
    (ARRAY[0.05,0.90,0.10,0.08]), (ARRAY[0.06,0.88,0.11,0.09]), (ARRAY[0.04,0.92,0.09,0.07]),
    (ARRAY[0.895,0.105,0.05,0.02]);
SELECT id,
       fela_cluster_over(VARIADIC embedding) OVER () AS theme,
       round(fela_anomaly_over(VARIADIC embedding) OVER ()::numeric, 3) AS novelty
FROM photos ORDER BY id;
SELECT s.support_idx AS photo_row, round(s.distance::numeric, 4) AS distance
FROM fela_similar(
       ARRAY[0.90,0.10,0.05,0.02]::float8[],
       (SELECT array_agg(v ORDER BY id, ord)
        FROM photos, LATERAL unnest(embedding) WITH ORDINALITY AS u(v, ord)),
       4, 3) AS s;

\echo '--- write-back example (imputation persisted) ---'
