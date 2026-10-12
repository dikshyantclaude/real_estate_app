# District Atlas

A complete Washington, DC real estate capstone: interactive analytics, geographic exploration, evaluated sale-price estimation, saved records, and content-based comparable-property recommendations.

## Run the application locally

No Node build or backend service is required. From this directory:

```bash
python -m http.server 8000 --directory dist
```

Open http://localhost:8000. Serve the files through HTTP rather than opening index.html directly; browsers restrict local file fetches.

## Reproduce training

```bash
python -m pip install -r requirements.txt
python scripts/train.py
```

The script repairs amenity extraction, joins land area using normalized SSL identifiers, derives age at sale, prepares chronological partitions, and compares a median-price baseline with Ridge, Random Forest, and Gradient Boosting. Regressors predict log sale price; outputs are converted back to dollars. Three expanding date-grouped validation windows choose the model. A final later-sale holdout is evaluated only after selection. Imputation, scaling, and one-hot encoding fit independently within each fold.

It exports the actual trained model into dist/model.json and verifies exported tree inference against sklearn predictions. dist/report.json contains real metrics, grouped tree importance, sample sizes, and limitations. data/inference_checks.json supports independent JavaScript inference parity tests.

## Structure

- dist/index.html, style.css, app.js: responsive application
- dist/vendor: locally bundled Leaflet 1.9.4; its JS/CSS retain the upstream notices
- dist/data.json, homes.csv: working property dataset
- dist/model.json: trained preprocessing parameters and model trees/coefficients
- dist/report.json: evaluation results
- scripts/train.py: reproducible Python data preparation, training, and export
- scripts/check-app.cjs: route-rendering and real inference parity checks using lightweight DOM stubs
- data/source.csv: user's PureClean.csv snapshot
- data/land_area.csv: CAMA land-area lookup
- data/nearby_provenance.json: category, distance, and source records for amenities

## Application modules

1. Overview: observed market distributions, neighborhood medians, area/price scatter.
2. Explore: address/place search, neighborhood/type/budget/bedroom filters, interactive map, saved properties, per-record detail, filtered CSV export.
3. Estimate: real exported trained-model inference with adjustable characteristics and coordinates, validation error band, and comparable records.
4. Comparables: hard preference filters plus similarity ranking by building area, rooms, type, and straight-line geographic distance. Amenity category and Metro distance filters use the stored enrichment provenance.
5. Model lab: CV and held-out regression metrics, grouped split importance, methodology, downloads.
6. Project & data: collection, scope, limitations, reproducibility, and sources.

## Important scope

- Recorded SALE prices, not asking prices or active listings. Recommendations are historical comparables, not purchase availability.
- Selected qualified single-unit residential properties; condos absent. Not a representative random sample of all DC real estate.
- Current building attributes may differ from historical-sale characteristics.
- Price per square foot is analytics only and excluded from prediction.
- Renovation year is excluded from prediction due to inconsistent/imputed source values.
- No mock gym, backyard, parking, furnishing, or luxury ratings were created.
- Plot area (land_sqft) differs from gross building area (sqft).
- Coordinates usually represent parcel centroids. Place distances are approximate straight-line distances, not walking times.
- The empirical validation-residual band is not a calibrated property-specific confidence interval.
- Nearby-place names/status come from DCGIS snapshots; current operation is not independently verified.
- Tree split importance is association, not causation, and can favor continuous features.

DCGIS sources: https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/ . OpenStreetMap base-map tiles require internet access; geographic point view is available if Leaflet fails to load. Map attribution: © OpenStreetMap contributors. No keys or paid APIs are needed for the app.

## Verification

Python model-export parity and JavaScript parity tests run on representative held-out records; independent route-rendering checks cover all six modules. Full browser QA was unavailable in the build environment. Live map tiles require an external network connection.
